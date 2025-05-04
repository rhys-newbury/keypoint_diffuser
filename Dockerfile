FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6+PTX"

# Install System Dependencies
RUN apt-get update && \
    apt-get install -y  software-properties-common && \
    add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt-get install -y git gcc g++ libegl1 libgl1 libgomp1 libgl1-mesa-glx libgl1-mesa-dri git rsync gcc-4.8 libstdc++6 libc6 p7zip-full python3-packaging

COPY requirements.txt requirements.txt

# Install pip  dependencies
RUN pip install --upgrade pip packaging && \
    FORCE_CUDA=1 pip install -vv -r requirements.txt

# Simpler to conda install these packages
RUN conda install pytorch-cluster pytorch-scatter pytorch-sparse -c pyg -y

WORKDIR /app

# Build EMD_loss
COPY src/keypoint_diffuser/utils/emd_loss /app/src/keypoint_diffuser/utils/emd_loss
RUN cd /app/src/keypoint_diffuser/utils/emd_loss && python3 setup.py install

RUN groupadd -g 1000 user && useradd -u 1000 -g 1000 -m user

COPY --chown=user:user . /app

# Install keypoint_diffuser
RUN cd /app && pip install -e .

WORKDIR app
USER user
