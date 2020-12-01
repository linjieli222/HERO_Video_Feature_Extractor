import torch as th
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import ffmpeg
import math
from data_utils import convert_to_float
from yuv_reader import read_yuv420p


class VideoLoader(Dataset):
    """Pytorch video loader."""

    def __init__(
            self,
            csv,
            preprocess,
            framerate=1,
            size=112,
            centercrop=False,
            pix_fmt='yuv420p',
            overwrite=False,
    ):
        """
        Args:
        """
        self.csv = pd.read_csv(csv)
        self.centercrop = centercrop
        self.size = size
        self.framerate = framerate
        self.preprocess = preprocess
        self.pix_fmt = pix_fmt
        self.overwrite = overwrite

    def __len__(self):
        return len(self.csv)

    def _get_video_info(self, video_path):
        probe = ffmpeg.probe(video_path)
        video_stream = next((stream for stream in probe['streams']
                             if stream['codec_type'] == 'video'), None)
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        fps = math.floor(convert_to_float(video_stream['avg_frame_rate']))
        try:
            frames_length = int(video_stream['nb_frames'])
            duration = float(video_stream['duration'])
        except Exception:
            frames_length, duration = -1, -1
        info = {"duration": duration, "frames_length": frames_length,
                "fps": fps, "height": height, "width": width}
        return info

    def _get_output_dim(self, h, w):
        if isinstance(self.size, tuple) and len(self.size) == 2:
            return self.size
        elif h >= w:
            return int(h * self.size / w), self.size
        else:
            return self.size, int(w * self.size / h)

    def __getitem__(self, idx):
        video_path = self.csv['video_path'].values[idx]
        output_file = self.csv['feature_path'].values[idx]
        load_flag = os.path.isfile(video_path)
        if not self.overwrite:
            load_flag = load_flag and not(os.path.isfile(output_file))
        if load_flag:
            # print('Decoding video: {}'.format(video_path))
            try:
                info = self._get_video_info(video_path)
                h, w = info["height"], info["width"]
            except Exception as e:
                print('ffprobe failed at: {}'.format(video_path))
                print(e)
                return {'video': th.zeros(1), 'input': video_path,
                        'output': output_file, 'info': {}}
            height, width = self._get_output_dim(h, w)
            cmd = (
                ffmpeg
                .input(video_path)
                .filter('fps', fps=self.framerate)
                .filter('scale', width, height)
            )
            if self.centercrop:
                x = int((width - self.size) / 2.0)
                y = int((height - self.size) / 2.0)
                cmd = cmd.crop(x, y, self.size, self.size)
            out, _ = (
                cmd.output('pipe:', format='rawvideo', pix_fmt=self.pix_fmt)
                .run(capture_stdout=True, quiet=True)
            )
            # 'rgb24'
            if self.centercrop and isinstance(self.size, int):
                height, width = self.size, self.size
            if self.pix_fmt == "rgb24":
                video = np.frombuffer(out, np.uint8).reshape(
                    [-1, height, width, 3])
                video = th.from_numpy(video)
            else:
                video = read_yuv420p(out, width, height)

            # video = video.permute(0, 3, 1, 2)
            video = self.preprocess(video, info)
            return {'video': video, 'input': video_path,
                    'output': output_file, 'info': info}

        else:
            video = th.zeros(1)
            return {'video': video, 'input': video_path, 'output': output_file,
                    'info': {}}


def pack_pathway_output(cfg, frames):
    """
    Prepare output as a list of tensors. Each tensor corresponding to a
    unique pathway.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is
            `batch` x `channel` x `num frames` x `height` x `width`.
    Returns:
        frame_list (list): list of tensors with the dimension of
            `batch` x `channel` x `num frames` x `height` x `width`.
    """
    if cfg.MODEL.ARCH in cfg.MODEL.SINGLE_PATHWAY_ARCH:
        frame_list = [frames]
    elif cfg.MODEL.ARCH in cfg.MODEL.MULTI_PATHWAY_ARCH:
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = th.index_select(
            frames,
            2,
            th.linspace(
                0, frames.shape[2] - 1, frames.shape[2] // cfg.SLOWFAST.ALPHA
            ).long().cuda(),
        )
        frame_list = [slow_pathway, fast_pathway]
    else:
        raise NotImplementedError(
            "Model arch {} is not in {}".format(
                cfg.MODEL.ARCH,
                cfg.MODEL.SINGLE_PATHWAY_ARCH + cfg.MODEL.MULTI_PATHWAY_ARCH,
            )
        )
    return frame_list


def clip_iterator(video, batch_size):
    n_chunk = len(video)
    n_iter = int(math.ceil(n_chunk / float(batch_size)))
    for i in range(n_iter):
        min_ind = i * batch_size
        max_ind = (i + 1) * batch_size
        # Transfer the data to the current GPU device.
        if isinstance(video, (list,)):
            inputs = []
            for i in range(len(video)):
                inputs.append(video[i][min_ind:max_ind])
        else:
            inputs = video[min_ind:max_ind]
        yield min_ind, max_ind, inputs
