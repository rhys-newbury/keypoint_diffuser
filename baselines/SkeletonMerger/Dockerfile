FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6+PTX"

# Install System Dependencies
RUN apt-get update && \
    apt-get install -y  software-properties-common && \
    add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt-get install -y git gcc g++ libegl1 libgl1 libgomp1 libgl1-mesa-glx libgl1-mesa-dri git rsync gcc-4.8 libstdc++6 libc6 p7zip-full python3-packaging

RUN pip install h5py simpleicp

RUN FORCE_CUDA=1 pip install -vv git+https://github.com/facebookresearch/pytorch3d.git@stable

RUN pip install einops "numpy<2"

RUN pip install open3d

COPY . .

