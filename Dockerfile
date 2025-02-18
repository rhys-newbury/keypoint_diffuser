FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

RUN pip install setuptools==69.5.1 iopath fvcore && \
    pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1110/download.html

WORKDIR /app

RUN pip install configargparse==1.4 tensorboardX einops open3d && \
    pip install --force-reinstall numpy==1.22.2 pandas==1.4.1 wandb tensorboard && \
    echo 'export PYTHONPATH="/app:${PYTHONPATH}"' >> ~/.bashrc

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-key del 7fa2af80
ADD https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb .
RUN dpkg -i cuda-keyring_1.0-1_all.deb

COPY keypointdeformer/PCT_Pytorch /app/keypointdeformer/PCT_Pytorch
RUN cd /app/keypointdeformer/PCT_Pytorch/pointnet2_ops_lib && pip install -e .

RUN echo 'export WANDB_API_KEY=db09fabd9a9cd7887ace1f168b3701bfa094f12b' >> ~/.bashrc

RUN apt-get update && apt-get install --no-install-recommends -y libegl1 libgl1 libgomp1

RUN apt-get install --reinstall -y libgl1-mesa-glx libgl1-mesa-dri git

RUN pip install trimesh ninja pre-commit
COPY . /app

WORKDIR app

RUN echo 'export PYTHONPATH=$PYTHONPATH:/app' >> ~/.bashrc
