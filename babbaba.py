import sys
import os
from tqdm import tqdm
import pathlib
import pandas as pd
events = list()
events.append({
            'vid_idx': 1,
            'is_start': True
        })
events.append({
            'vid_idx': 2,
            'is_start': True
        })
print(events)