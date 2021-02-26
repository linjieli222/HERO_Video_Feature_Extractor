import torch as th
import numpy as np
from video_loader import (
    VideoLoader, clip_iterator)
from video_dataflow import VideoDataFlow, ReadVideo
from dataflow import MultiProcessMapDataZMQ
from torch.utils.data import DataLoader
import argparse
from model import build_model
from preprocessing import Preprocessing
from random_sequence_shuffler import RandomSequenceSampler
from tqdm import tqdm
from prefetch_loader import PrefetchLoader
import os
import time
from yuv_reader import YuvRgbConverter
FEATURE_LENGTH = 1024
YUV2RGB = YuvRgbConverter()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Easy video feature extractor')

    parser.add_argument(
        '--csv',
        type=str, default='extract_feature/tmp.csv',
        help='input csv with video input path')
    parser.add_argument(
            "--s3d_ckpt",
            help="Path to the pretrained s3d",
            default="/models/s3d_howto100m.pth",
            type=str,
        )
    parser.add_argument(
        '--batch_size', type=int, default=4, help='batch size')
    parser.add_argument(
        '--half_precision', type=int, default=1,
        help='output half precision float')
    parser.add_argument(
        '--dataflow', action='store_true',
        help='enable dataflow')
    parser.add_argument(
        '--overwrite', action='store_true',
        help='allow overwrite output files')
    parser.add_argument(
        '--num_decoding_thread', type=int, default=0,
        help='Num parallel thread for video decoding')
    parser.add_argument(
        '--target_framerate', type=int, default=30,
        help='decoding frame per second')
    parser.add_argument(
        '--clip_len', type=str, default='3/2',
        help='decoding length of clip (in seconds)')
    parser.add_argument(
        '--pix_fmt', type=str, default="rgb24", choices=["rgb24", "yuv420p"],
        help='decode video into RGB format')
    return parser.parse_args()


@th.no_grad()
def perform_test(test_loader, model, args, failed_log, n_dataset):
    """
    For classification:
    Perform mutli-view testing that uniformly samples
        N clips from a video along
    its temporal axis. For each clip, it takes 3 crops to cover the spatial
    dimension, followed by averaging the softmax scores across all Nx3 views to
    form a video-level prediction. All video predictions are compared to
    ground-truth labels and the final testing performance is logged.
    For detection:
    Perform fully-convolutional testing on the full frames without crop.
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """

    # Enable eval mode.
    model.eval()
    totatl_num_frames = 0

    total_time = 0
    pbar = tqdm(total=n_dataset)
    for _, data in enumerate(test_loader):
        video = data['video']
        video_shape_len = len(video.shape)
        input_file = data['input']
        output_file = data['output']
        if isinstance(input_file, (list,)):
            input_file = input_file[0]
            output_file = output_file[0]
        # print(f"Processing {input_file}")
        if video_shape_len == 6:
            video = video.squeeze(0)
        video_shape_len = len(video.shape)
        if video_shape_len == 5:
            n_chunk = len(video)
            features = th.cuda.HalfTensor(
                n_chunk, FEATURE_LENGTH).fill_(0)
            clip_loader = PrefetchLoader(clip_iterator(video, args.batch_size))

            for _, (min_ind, max_ind, inputs) in enumerate(clip_loader):
                # B T H W C
                inputs = inputs.float()
                if args.pix_fmt == "yuv420p":
                    inputs = YUV2RGB(inputs)
                inputs = inputs.permute(0, 4, 1, 2, 3)
                # -> B C T H W
                inputs = inputs/255.
                # Perform the forward pass.
                th.cuda.synchronize()
                start_time = time.time()
                batch_features = model(inputs)
                th.cuda.synchronize()
                end_time = time.time()
                total_time += end_time - start_time
                _, action_emb = (
                    batch_features["video_embedding"],
                    batch_features["mixed_5c"])
                features[min_ind:max_ind] = action_emb.half()
            features = features.cpu().numpy()
            features = features.astype('float16')
            totatl_num_frames += features.shape[0]

            # safeguard output path before saving
            dirname = os.path.dirname(output_file)
            if not os.path.exists(dirname):
                print(f"Output directory {dirname} does not exists" +
                      ", creating...")
                os.makedirs(dirname)
            try:
                # np.savez_compressed(output_file, features=features)
                np.savez(output_file, features=features)
            except Exception as e:
                print(e)
                print(output_file)
        elif os.path.isfile(output_file):
            print(f'Video {input_file} already processed.')
        elif not os.path.isfile(input_file):
            failed_log.write(f'{input_file}, does not exist.\n')
        else:
            failed_log.write(f'{input_file}, failed at ffprobe.\n')
        pbar.update(1)

    print(f"Total number of frames: {totatl_num_frames}")
    print(f"Model inference time: {total_time}")


def main():
    """
    Main function to extract features.
    """
    args = parse_args()
    failed_log = open(args.csv.split(".csv")[0]+"_failed.txt", "w")

    preprocess = Preprocessing(
        target_fps=args.target_framerate,
        size=224, clip_len=args.clip_len, padding_mode='tile')
    if args.dataflow:
        readvideo = ReadVideo(
            preprocess,
            framerate=args.target_framerate,
            size=224,
            centercrop=True,
            pix_fmt=args.pix_fmt,
            overwrite=args.overwrite
        )
        dataset = VideoDataFlow(args.csv)
        loader = MultiProcessMapDataZMQ(
            dataset, num_proc=args.num_decoding_thread,
            map_func=readvideo, strict=True)
        loader.reset_state()
        n_dataset = len(dataset)
    else:
        dataset = VideoLoader(
            args.csv,
            preprocess,
            framerate=args.target_framerate,
            size=224,
            centercrop=True,
            pix_fmt=args.pix_fmt,
            overwrite=args.overwrite
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

    model = build_model(args)
    perform_test(
        loader, model, args, failed_log, n_dataset)


if __name__ == "__main__":
    main()
