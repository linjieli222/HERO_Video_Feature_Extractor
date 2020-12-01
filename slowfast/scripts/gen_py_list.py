# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import json
import numpy as np
import os
import shutil
from tqdm import tqdm

srclist = './data/kinetics/kinetics400/classids.json'

# videodir = 'YOUR_DATASET_FOLDER/train/'
# outlist = 'trainlist.txt'

videodir = './data/kinetics/kinetics400/videos/valid/'
outlist = './data/kinetics/kinetics400/testlist.txt'



f = open(outlist, 'w')


json_data = open(srclist).read()
clss_ids = json.loads(json_data)

folder_list = os.listdir(videodir)

was = np.zeros(1000)

for i in tqdm(range(len(folder_list))):

    folder_name = folder_list[i]
    folder_name_v1 = videodir + folder_name

    names = folder_name.split(' ')
    
    newfolder_name = ''
    for j in range(len(names)):
        newfolder_name = newfolder_name + names[j]
        if j < len(names) - 1:
            newfolder_name = newfolder_name + '_'
    folder_name_cmb = videodir + newfolder_name

    if folder_name_v1 != folder_name_cmb:
        print(folder_name, newfolder_name)
        shutil.move(folder_name_v1, folder_name_cmb)
    
    class_label = folder_name.split('_')
    if len(class_label) > 1:
        class_label = '\"' +" ".join(class_label) + '\"'
    else:
        class_label = class_label[0]

    lbl = clss_ids[class_label]
    
    was[lbl] = 1

    video_list = os.listdir(folder_name_cmb)

    for j in range(len(video_list)):
        video_file = video_list[j]
        if "raw." in video_file:
            print( folder_name_cmb + '/' + video_file)
            continue
        video_file = folder_name_cmb + '/' + video_file
        f.write(video_file + ' ' + str(lbl) + '\n')


f.close()

cnt = 0
for i in range(400):
    cnt = cnt + was[i]

print(cnt)

