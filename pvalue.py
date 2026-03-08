import os
import sys
import copy
import time
import torch
import torch.nn as nn  # 需要 nn.CrossEntropyLoss
import pickle
import random
import pathlib
import torchvision
import pandas as pd
import numpy as np
from collections import Counter
from torchvision import datasets, transforms as T
import configparser
import itertools
from scipy import stats  # 用于统计分析
from sklearn import linear_model, model_selection  # 用于 MIA

def calculate_statistics(current_values, baseline_value):
    """计算统计指标: Mean, Std, P-value, Cohen's d, Cliff's delta"""
    values = np.array(current_values)
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    
    # P-Value
    # if std < 1e-9:
    #     p_val = 0.0 if abs(mean - baseline_value) > 1e-5 else 1.0
    # else:
    t_stat, p_val = stats.ttest_1samp(values, baseline_value)

    print(f"t_stat: {t_stat}, p_val: {p_val}")

if __name__ == "__main__":
    current_values = [0.36, 0.28, 0.42, 0.36, 0.34]
    baseline_value = 0.0
    calculate_statistics(current_values, baseline_value)
