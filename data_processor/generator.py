import argparse

import pandas as pd
from concurrent.futures import *

from utils import *
from exceptions import *


class PathEntity:
    def __init__(self, data_path_2d, data_path_3d, no):
        self.data_path_2d = data_path_2d
        self.data_path_3d = data_path_3d
        self.no = no


def cmp_func(ele):
    return ele.no


def generate():
    path = os.getcwd()

    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", "-f", default="for_chuangxinshijian")
    args = parser.parse_args()

    data_dir = None
    for filename in os.listdir(path):
        if filename == args.filename:
            data_dir = os.listdir(os.path.join(path, filename))

    if data_dir is None:
        raise DataDirNotDetectedError("The directory for the data doesn't detect in the current directory!")

    ignore_dirs = ['cf', 'log']
    data_dir = [os.path.join(path, args.filename, sub_dir) for sub_dir in data_dir if sub_dir not in ignore_dirs]

    paths = []
    tif_data_paths, volume_data_paths = [], []
    with ThreadPoolExecutor(max_workers=40) as t:
        obj_list = []
        for sub_dir in data_dir:
            obj = t.submit(generate_data_path, sub_dir)
            obj_list.append(obj)

        for obj in as_completed(obj_list):
            result = obj.result()
            paths.append(PathEntity(result[0], result[1], result[2]))

    paths.sort(key=cmp_func)
    for p in paths:
        tif_data_paths.append(p.data_path_2d)
        volume_data_paths.append(p.data_path_3d)
    data = {"2D_data_path": tif_data_paths, "3D_data_path": volume_data_paths}
    df = pd.DataFrame(data)
    data_path = os.path.join(path, "data_path.csv")
    df.to_csv(data_path, index=False)
    print("Successfully Generating the data!")
