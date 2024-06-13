import sys
import os
from tqdm import tqdm
import pathlib
import pandas as pd
import numpy as np

# 假设 nobs 是一个包含多个观测数组的字典
nobs = {
    
    'obs2': np.array([[5, 6]]),            # 形状为 (1, 2)
    'obs1': np.array([[1, 2], [3, 4]])     # 形状为 (2, 2)
}

# 提取第一个观测数组的批量大小
B = next(iter(nobs.values())).shape[0]
print(B)  # 输出: 2