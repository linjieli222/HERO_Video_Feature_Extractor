FROM nvcr.io/nvidia/pytorch:20.03-py3

COPY requirement.txt ./
RUN pip install -r requirement.txt && rm ./requirement.txt

# additional python dependencies
RUN pip install 'git+https://github.com/facebookresearch/fvcore' \
    'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

# Install av and ffmpeg 
RUN apt-get update
RUN apt-get install -y libavformat-dev libavcodec-dev libavdevice-dev \
   libavutil-dev libswscale-dev libswresample-dev libavfilter-dev \
   autoconf automake build-essential cmake \
   libass-dev libfreetype6-dev libjpeg-dev libtheora-dev \
   libtool libvorbis-dev libx264-dev \
   pkg-config wget yasm zlib1g-dev libpq-dev
RUN pip install av==6.2.0
RUN apt-get install ffmpeg 
RUN pip install ffmpeg-python

RUN pip install --upgrade git+https://github.com/tensorpack/dataflow.git

#### detectron 2 installation
RUN git clone https://github.com/facebookresearch/detectron2 /detectron2_repo
ENV FORCE_CUDA="1"
ENV TORCH_CUDA_ARCH_LIST="Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
RUN pip install -e /detectron2_repo


# add new command here
WORKDIR /src

RUN mkdir /models && \
    wget https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/SLOWFAST_8x8_R50.pkl \
        -O /models/SLOWFAST_8x8_R50.pkl
Run wget https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/s3d_howto100m.pth \
        -O /models/s3d_howto100m.pth