import sys

import torch
import numpy as np
import cv2
import ffmpeg


def read_y(w, h, binary):
    x = np.frombuffer(binary, dtype=np.uint8).reshape((h, w))
    return torch.from_numpy(x).float()


def read_uv(w, h, binary):
    x = np.frombuffer(binary, dtype=np.uint8
                      ).reshape((h//2, w//2))
    x = cv2.resize(x, (w, h), cv2.INTER_NEAREST)
    return torch.from_numpy(x).float()


def read_yuv420p(binary, w, h):
    assert w % 2 == h % 2 == 0
    tot_len = len(binary)
    frame_length = w*h*6//4
    n_frame = tot_len // frame_length
    n_pix = w*h
    uv_len = n_pix // 4
    yuv = torch.Tensor(n_frame, h, w, 3)

    y_starts = range(0, tot_len, frame_length)
    for i, st in enumerate(y_starts):
        yuv.data[i, ..., 0] = read_y(w, h, binary[st: st+n_pix]).data

    u_starts = range(n_pix, tot_len, frame_length)
    for i, st in enumerate(u_starts):
        yuv.data[i, ..., 1] = read_uv(w, h, binary[st: st+uv_len]).data

    v_starts = range(n_pix+uv_len, tot_len, frame_length)
    for i, st in enumerate(v_starts):
        yuv.data[i, ..., 2] = read_uv(w, h, binary[st: st+uv_len]).data

    return yuv


class YuvRgbConverter(object):
    def __init__(self, device=torch.device('cuda')):
        self.T = torch.tensor(
            [[1.164,  1.164, 1.164],
             [0,     -0.392, 2.017],
             [1.596, -0.813, 0]],
            requires_grad=False
        ).to(device)
        self.bias = torch.tensor([-16, -128, -128],
                                 requires_grad=False).float().to(device)

    @torch.no_grad()
    def __call__(self, yuv):
        rgb = (yuv+self.bias).matmul(self.T).detach().clamp_(0, 255)
        return rgb


YUV2RGB = YuvRgbConverter()


def _get_video_dim(video_path):
    probe = ffmpeg.probe(video_path)
    video_stream = next((stream for stream in probe['streams']
                         if stream['codec_type'] == 'video'), None)
    width = int(video_stream['width'])
    height = int(video_stream['height'])
    return height, width


def read_video_rgb(video_path):
    h, w = _get_video_dim(video_path)
    cmd = ffmpeg.input(video_path)
    out, _ = (
        cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .run(capture_stdout=True, quiet=True)
    )
    video = np.frombuffer(out, np.uint8).reshape([-1, h, w, 3])
    video = torch.from_numpy(video).float().cuda()
    torch.cuda.synchronize()
    return video


def read_video_yuv(video_path):
    h, w = _get_video_dim(video_path)
    cmd = ffmpeg.input(video_path)
    out, _ = (
        cmd.output('pipe:', format='rawvideo', pix_fmt='yuv420p')
        .run(capture_stdout=True, quiet=True)
    )
    video = read_yuv420p(out, w, h).float().cuda()
    video = YUV2RGB(video)
    torch.cuda.synchronize()
    return video


def test(video_path):
    rgb_v = read_video_rgb(video_path)
    cv2.imwrite('rgb.png', rgb_v[0].cpu().numpy())
    yuv_v = read_video_yuv(video_path)
    cv2.imwrite('yuv.png', yuv_v[0].cpu().numpy())


if __name__ == '__main__':
    path = sys.argv[1]
    test(path)
