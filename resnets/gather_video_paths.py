import os

videopath = "/video/"
feature_path = "/output/resnet_features"
csv_folder = "/output/csv"
if not os.path.exists(csv_folder):
    os.mkdir(csv_folder)
if not os.path.exists(feature_path):
    os.mkdir(feature_path)

outputFile = f"{csv_folder}/resnet_info.csv"
with open(outputFile, "w") as fw:
    fw.write("video_path,feature_path\n")
    fileList = [f for f in os.listdir(videopath)
                if os.path.isfile(os.path.join(videopath, f))]

    for filename in fileList:
        input_filename = os.path.join(videopath, filename)
        fileId, file_extension = os.path.splitext(filename)
        output_filename = os.path.join(
            feature_path, fileId+".npz")
        fw.write(input_filename+","+output_filename+"\n")
