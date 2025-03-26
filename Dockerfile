# FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6+PTX"

RUN apt update && apt install -y git gcc g++
RUN FORCE_CUDA=1 pip install -vv "git+https://github.com/facebookresearch/pytorch3d.git@stable"

RUN pip install setuptools==69.5.1 iopath fvcore && \
    pip install pytorch3d
    # -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu117_pyt1110/download.html

WORKDIR /app

RUN pip install configargparse==1.4 tensorboardX einops open3d && \
    pip install --force-reinstall numpy==1.22.2 pandas==1.4.1 wandb tensorboard matplotlib==3.5.0 && \
    echo 'export PYTHONPATH="/app:${PYTHONPATH}"' >> ~/.bashrc

# RUN rm /etc/apt/sources.list.d/cuda.list
# RUN rm /etc/apt/sources.list.d/nvidia-ml.list

# RUN apt-key del 7fa2af80
# ADD https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb .
# RUN dpkg -i cuda-keyring_1.0-1_all.deb
# ENV CUDA_HOME="/usr/local/cuda"

COPY keypointdeformer/PCT_Pytorch /app/keypointdeformer/PCT_Pytorch
RUN cd /app/keypointdeformer/PCT_Pytorch/pointnet2_ops_lib && pip install -e .

RUN echo 'export WANDB_API_KEY=db09fabd9a9cd7887ace1f168b3701bfa094f12b' >> ~/.bashrc

RUN apt-get update && apt-get install --no-install-recommends -y libegl1 libgl1 libgomp1

RUN apt-get install --reinstall -y libgl1-mesa-glx libgl1-mesa-dri git rsync

RUN pip install trimesh ninja pre-commit
RUN pip install flash-attn==2.3.2 timm spconv-cu116
RUN conda install pytorch-cluster pytorch-scatter pytorch-sparse -c pyg -y

RUN apt install -y p7zip-full software-properties-common

RUN add-apt-repository ppa:ubuntu-toolchain-r/test

RUN apt-get update

RUN apt-get install -y gcc-4.8

RUN apt-get upgrade -y libstdc++6
RUN apt install -y libc6
COPY . /app
WORKDIR app

RUN echo 'export PYTHONPATH=$PYTHONPATH:/app' >> ~/.bashrc
RUN pip install pymeshlab rtree
