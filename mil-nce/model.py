import torch as th
from s3dg import S3D


def build_model(args):
    print(f'Loading S3D with checkpoint {args.s3d_ckpt}...')
    model = S3D()
    model = model.cuda()
    model_data = th.load(args.s3d_ckpt)
    model.load_state_dict(model_data, strict=False)

    model.eval()
    return model
