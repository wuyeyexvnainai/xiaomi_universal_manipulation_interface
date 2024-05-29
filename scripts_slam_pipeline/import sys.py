import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
print(os.path.dirname(__file__))
print(ROOT_DIR)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)
