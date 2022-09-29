import argparse
import os.path
from concurrent.futures import *

from utils import *
from exceptions import *

path = os.getcwd()
parent_path = os.path.dirname(path)

parser = argparse.ArgumentParser()
parser.add_argument("--filename", "-f", default="for_chuangxinshijian")
args = parser.parse_args()

data_dir = None
for filename in os.listdir(parent_path):
    if filename == args.filename:
        data_dir = os.listdir(os.path.join(parent_path, filename))

if data_dir is None:
    raise DataDirNotDetectedError("The directory for the data doesn't detect in the current directory!")

ignore_dirs = ['cf', 'log']
data_dir = [os.path.join(parent_path, args.filename, sub_dir) for sub_dir in data_dir if sub_dir not in ignore_dirs]

with ThreadPoolExecutor(max_workers=40) as t:
    obj_list = []
    for sub_dir in data_dir:
        obj = t.submit(clean_data_npy, sub_dir)
        obj_list.append(obj)
    for obj in as_completed(obj_list):
        result = obj.result()

print("Successfully delete all volume.npy files!")
