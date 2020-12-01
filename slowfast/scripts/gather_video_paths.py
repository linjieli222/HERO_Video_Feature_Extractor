import os
from os import walk
import json
from tqdm import tqdm

outputFile = "./data/kinetics/kinetics400/test.csv"
mypath = "./data/kinetics/kinetics400/videos/valid_256/"
clssids = json.load(open("./data/kinetics/kinetics400/classids.json", "r"))

fileList = []
for (dirpath, dirnames, filenames) in walk(mypath):
    for filename in filenames:
        fileList.append((dirpath,filename))

with open(outputFile, "w") as fw:
    # fw.write("video_path,feature_path\n")
    for (dirpath, filename) in tqdm(fileList):
        input_filename = os.path.join(dirpath, filename)
        label = dirpath.split("/")[-1]
        if "_" in label:
            label = " ".join(label.split("_"))
            label = '"'+label+'"'
        label_id = clssids[label]
        fw.write(input_filename+" "+str(label_id)+"\n")
