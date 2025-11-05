#!/usr/bin/env python
# coding=utf-8
"""
@author: xiaxu
@license: MIT
@file: config.py
@date: 2025/10/23 15:09
@desc:基本的配置路径，超参数
"""
from pathlib import Path


ROOT_DIR = Path(__file__).parent.parent # 根目录
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
LOGS_DIR = ROOT_DIR / "logs"
MODELS_DIR = ROOT_DIR / "models"

#模型设置
BATCH_SIZE = 64  #64过大会退出
DIM_MODEL = 128 #d_model 特征维度
N_HEADS = 8
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2


# 超参数设置
MAX_SEQ_LENG = 128  #最大长度
LEARNING_RATE = 1e-3
EPOCHS = 20

