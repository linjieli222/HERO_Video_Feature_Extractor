import torch
import numpy as np
import cv2


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

