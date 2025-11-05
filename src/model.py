#!/usr/bin/env python
# coding=utf-8
"""
@author: xiaxu
@license: MIT
@file: model.py
@date: 2025/10/27 10:18
@desc: seq2seq模型
"""
import math

import torch
from torch import nn
from src import config


#位置编码
class PositionalEncoding(nn.Module):
    """
    生成位置编码[pos_size,d_model]
    """
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros([max_len, d_model]) #初始化位置编码
        for pos in range(max_len): #生成每个位置的编码
            for _2i in range(0,d_model,2): #每个位置生成d_mode长的编码
                pe[pos, _2i] = math.sin(pos / (10000 ** (_2i / d_model)))
                pe[pos, _2i+1] = math.cos(pos / (10000 ** (_2i / d_model)))
        self.register_buffer('pe', pe)
        # 添加一个缓存到moduel,不作为参数，持久保存到moduel

    def forward(self, x):
        # x.shape:[B,S,D]
        seq_len = x.shape[1]
        part_pe = self.pe[0:seq_len]
        # part_pe:[seq_len,dim_model]
        return x+part_pe


class TranslationModel(nn.Module):
    def __init__(self,zh_vocab_size,en_vocab_size,zh_padding_index,en_padding_index):
        super().__init__()
        self.zh_embedding = nn.Embedding(zh_vocab_size, config.DIM_MODEL,padding_idx=zh_padding_index)
        self.en_embedding = nn.Embedding(en_vocab_size, config.DIM_MODEL,padding_idx=en_padding_index)

        #位置编码
        self.pos = PositionalEncoding(config.DIM_MODEL, config.MAX_SEQ_LENG)

        self.transformer = nn.Transformer(d_model=config.DIM_MODEL,
                                          nhead=config.N_HEADS,
                                          num_encoder_layers=config.NUM_ENCODER_LAYERS,
                                          num_decoder_layers=config.NUM_DECODER_LAYERS,
                                          batch_first=True)

        self.linear = nn.Linear(config.DIM_MODEL, en_vocab_size) #输出词的概率预测

    def forward(self, src,tgt,src_padding_mask,tgt_mask):
        memory = self.encode(src,src_padding_mask)
        return self.decode(tgt,memory,tgt_mask,src_padding_mask)

    def encode(self, src, src_pad_mask):
        """
        编码器构造，接收src,pad，输出memory，供decoder用
        :param src [batch_size, src_len]
        :param src_pad_mask [batch_size, src_len]:掩蔽pad[flase，True]true表示pad
        :return: [batch_size, seq_len, d_model]
        """
        embedded = self.zh_embedding(src)
        embedded = self.pos(embedded) # +位置编码
        return self.transformer.forward(src=embedded, src_key_padding_mask=src_pad_mask)

    def decode(self, tgt, memory, tgt_mask, momory_pad_mask):
        """
        解码器构造，接收memory和tgt,
        :param tgt [batch_size, tgt_len]
        :param memory [batch_size, src_len, d_model]
        :param tgt_mask [tgt_len, tgt_len] 遮掩tgt,防止看到后面的信息，[0,inf]
        :param momory_pad_mask [batch_size, src_len]
        :return: [batch_size, tgt_len, d_model]
        """
        embedded = self.en_embedding(tgt)
        embedded = self.pos(embedded)

        output = self.transformer.decoder(tgt=embedded, memory=memory,
                                          tgt_mask=tgt_mask,memory_mask=momory_pad_mask)
        outputs = self.linear(output)

        return outputs


