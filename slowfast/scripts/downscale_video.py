# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import fnmatch
import glob
import json
import os
import shutil
import subprocess
import uuid

from joblib import delayed
from joblib import Parallel
import pandas as pd
from tqdm import tqdm

# file_src = 'trainlist.txt'
# folder_path = 'YOUR_DATASET_FOLDER/train/'
# output_path = 'YOUR_DATASET_FOLDER/train_256/'


file_src = './data/kinetics/kinetics400/testlist.txt'
folder_path = './data/kinetics/kinetics400/videos/valid/'
output_path = './data/kinetics/kinetics400/videos/valid_256/'


file_list = []
dir_list = set()
f = open(file_src, 'r')

for line in f:
    rows = line.split()
    fname = rows[0]
    nameset  = fname.split('/')
    videoname = nameset[-1]
    classname = nameset[-2]

    output_folder = output_path + classname
    dir_list.add(output_folder)
    file_list.append(fname)

f.close()


def downscale_clip(inname, outname):

    status = False
    inname = '"%s"' % inname
    outname = '"%s"' % outname
    # -y allow overwrite exsiting files
    command = "ffmpeg -y -loglevel panic -i {} -filter:v scale=\"trunc(oh*a/2)*2:256\" -q:v 1 -c:a copy {}".format( inname, outname)
    try:
        output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as err:
        print(status, err.output, outname)
        return status, err.output
   
    status = os.path.exists(outname)
    if status:
        out_info = os.stat(outname)
        if out_info.st_size < 1000:
            in_info = os.stat(inname)
            print(f"Downscale error: {in_info.st_size} vs. {out_info.st_size}")
            print(f">>>>>>>> {outname}")
    # else:
    #     print(f"No such output file: {outname}")
    return status, 'Downscaled'


def downscale_clip_wrapper(row):

    nameset  = row.split('/')
    videoname = nameset[-1]
    classname = nameset[-2]

    output_folder = output_path + classname
    if os.path.isdir(output_folder) is False:
        try:
            os.mkdir(output_folder)
        except:
            print(output_folder)

    inname = folder_path + classname + '/' + videoname
    outname = output_path + classname + '/' + videoname

    downscaled = False
    if os.path.exists(outname):
        out_info = os.stat(outname)
        if out_info.st_size < 1000:
            in_info = os.stat(inname)
            print(f"Previous downscale error: {in_info.st_size} vs. {out_info.st_size}")
            print(f">>>>>>>> {inname}")
            downscaled, _ = downscale_clip(inname, outname)
    else:
        downscaled, _ = downscale_clip(inname, outname)
    return downscaled


for output_folder in tqdm(dir_list):
    if os.path.isdir(output_folder) is False:
        try:
            os.mkdir(output_folder)
        except:
            print("Error in creating", output_folder)
status_lst = Parallel(n_jobs=40)(delayed(downscale_clip_wrapper)(row) for row in tqdm(file_list))
