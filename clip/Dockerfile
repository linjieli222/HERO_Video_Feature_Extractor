FROM nvcr.io/nvidia/pytorch:20.09-py3

COPY requirement.txt ./
RUN pip install -r requirement.txt && rm ./requirement.txt


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

RUN pip install git+https://github.com/openai/CLIP.git

# add new command here
WORKDIR /src
