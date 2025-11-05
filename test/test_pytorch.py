#!/usr/bin/env python
# coding=utf-8
"""
@author: xiaxu
@license: MIT
@file: test_pytorch.py
@date: 2025/11/5 11:47
@desc: 
"""
import torch


torch.manual_seed(20)
test = torch.randn(2,3,4)
print(test)
print(test.shape[1])
print(test.reshape(-1)) #[] 沿着-1轴展成一维
print(test.reshape(-1,test.shape[-1])) #[ ,4]