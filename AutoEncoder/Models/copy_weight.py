from pathlib import Path
import shutil

path_in = 'saved_weight/AE_min2223'
path_out = '../operate_min/model_weight'

for dir_in in Path(path_in).iterdir():
    sid = dir_in.stem[5:]
    dir_out = Path(path_out)/sid
    shutil.copytree(dir_in, dir_out)