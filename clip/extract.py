import torch as th
import math
import numpy as np
from video_loader import VideoLoader
from torch.utils.data import DataLoader
import argparse
from preprocessing import Preprocessing
from random_sequence_shuffler import RandomSequenceSampler
import torch.nn.functional as F
from tqdm import tqdm
import os
import clip


parser = argparse.ArgumentParser(description='Easy video feature extractor')

parser.add_argument(
    '--csv',
    type=str,
    help='input csv with video input path')
parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size')
parser.add_argument(
        '--clip_len', type=float, default=3/2,
        help='decoding length of clip (in seconds)')
parser.add_argument(
        '--overwrite', action='store_true',
        help='allow overwrite output files')
parser.add_argument('--half_precision', type=int, default=1,
                    help='output half precision float')
parser.add_argument('--num_decoding_thread', type=int, default=4,
                    help='Num parallel thread for video decoding')
parser.add_argument('--model_version', type=str, default="ViT-B/32",
                    choices=["ViT-B/32", "RN50x4"],
                    help='Num parallel thread for video decoding')
args = parser.parse_args()

# model_version = "RN50x4" # "RN50x4"  # "ViT-B/32"
output_feat_size = 512 if args.model_version == "ViT-B/32" else 640
dataset = VideoLoader(
    args.csv,
    framerate=1/args.clip_len,
    size=224 if args.model_version == "ViT-B/32" else 288,
    centercrop=True,
    overwrite=args.overwrite,
    model_version=args.model_version
)
n_dataset = len(dataset)
sampler = RandomSequenceSampler(n_dataset, 10)
loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=args.num_decoding_thread,
    sampler=sampler if n_dataset > 10 else None,
)
preprocess = Preprocessing()
model, _ = clip.load(args.model_version, device="cuda")

totatl_num_frames = 0
with th.no_grad():
    for k, data in enumerate(tqdm(loader)):
        input_file = data['input'][0]
        output_file = data['output'][0]
        if args.model_version == "ViT-B/32":
            output_file = output_file.replace(
                "clip_features", "clip-vit_features")
        elif args.model_version == "RN50x4":
            output_file = output_file.replace(
                "clip_features", "clip-rn50x4_features")
        if os.path.isfile(output_file):
            # print(f'Video {input_file} already processed.')
            continue
        elif not os.path.isfile(input_file):
            print(f'{input_file}, does not exist.\n')
        elif len(data['video'].shape) > 4:
            video = data['video'].squeeze(0)
            if len(video.shape) == 4:
                video = preprocess(video)
                n_chunk = len(video)
                features = th.cuda.FloatTensor(
                    n_chunk, output_feat_size).fill_(0)
                n_iter = int(math.ceil(n_chunk / float(args.batch_size)))
                for i in range(n_iter):
                    min_ind = i * args.batch_size
                    max_ind = (i + 1) * args.batch_size
                    video_batch = video[min_ind:max_ind].cuda()
                    batch_features = model.encode_image(video_batch)
                    features[min_ind:max_ind] = batch_features
                features = features.cpu().numpy()
                if args.half_precision:
                    features = features.astype('float16')
                totatl_num_frames += features.shape[0]
                # safeguard output path before saving
                dirname = os.path.dirname(output_file)
                if not os.path.exists(dirname):
                    print(f"Output directory {dirname} does not exists, creating...")
                    os.makedirs(dirname)
                np.savez(output_file, features=features)
        else:
            print(f'{input_file}, failed at ffprobe.\n')

print(f"Total number of frames: {totatl_num_frames}")
