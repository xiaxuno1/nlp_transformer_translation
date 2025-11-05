#!/usr/bin/env python
# coding=utf-8
"""
@author: xiaxu
@license: MIT
@file: train.py
@date: 2025/10/27 11:25
@desc: 
"""
import time

import torch
from tqdm import tqdm

from src.tokenizer import ChineseTokenizer,EnglishTokenizer
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from src import config
from src.dataset import get_dataloader
from src.model import TranslationModel


def train_one_epoch(model, optimizer, train_dataloader, loss_fn, device):
    model.train()
    total_loss = 0
    for inputs_tensor,targets_tensor in tqdm(train_dataloader,desc="train"):
        encoder_inputs = inputs_tensor.to(device) #编码器的输入
        targets = targets_tensor.to(device) #[batch_size,seq_len]
        # print(encoder_inputs.shape, targets.shape)
        decoder_inputs = targets[:,:-1] # 去掉最后一个输入[batch_size,targets]；teacher forcing中输入<BOS>, 我, 爱, 你
        decoder_targets = targets[:,1:] # 去掉第一个输入；输出预测                      我, 爱, 你, <EOS>
        #decoder_inputs decoder_targets：[batch_size,seq_len]

        #forward
        src_pad_mask = (encoder_inputs == model.zh_embedding.padding_idx) #[B,src_len]  one:[flase,true]
        tgt_mask = model.transformer.generate_square_subsequent_mask(decoder_inputs.shape[1]) # [B,tgt_len] one:[0,inf]
        decoder_outputs = model(encoder_inputs, decoder_inputs, src_pad_mask,tgt_mask) #forward 前向传播
        # decoder_outputs：[batch_size,seq_len,en_vocab_size]
        # crossEntropyloss(outputs:[B*S,vocab_size],targets:B*S)
        loss = loss_fn(decoder_outputs.reshape(-1,decoder_outputs.shape[-1]), decoder_targets.reshape(-1))

        #反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / len(train_dataloader)

def tarin():
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #dataloader
    dataloader = get_dataloader()

    # 加载tokenizer
    en_tokenizer = EnglishTokenizer.from_vocab(config.MODELS_DIR / 'en_vocab.txt')
    zh_tokenize = ChineseTokenizer.from_vocab(config.MODELS_DIR / 'zh_vocab.txt')

    #定义模型
    model = TranslationModel(zh_tokenize.vocab_size, en_tokenizer.vocab_size,
                               zh_tokenize.pad_token_index, en_tokenizer.pad_token_index).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    # 定义loss
    loss_fn = nn.CrossEntropyLoss(ignore_index=en_tokenizer.pad_token_index).to(device) #计算loss时忽略pad填充

    # 定义optim
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    #tensorboard
    writer = SummaryWriter(log_dir=config.LOGS_DIR / time.strftime('%Y-%m-%d_%H-%M-%S'))

    best_loss = float('inf')
    for epoch in range(config.EPOCHS):
        print(f'====== epoch: {epoch+1}/{config.EPOCHS}  =====')
        loss = train_one_epoch(model,optimizer,dataloader,loss_fn,device)
        print(f'loss: {loss:4f}')

        writer.add_scalar('loss', loss, epoch+1)

        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), config.MODELS_DIR / 'best_model.pt')
            print(f'save best loss success: {best_loss:4f}')
    writer.close()



if __name__ == '__main__':
    tarin()