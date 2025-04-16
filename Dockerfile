# FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6+PTX"

RUN apt-get update && \
    apt-get install -y  software-properties-common && \
    add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt-get install -y git gcc g++ libegl1 libgl1 libgomp1 libgl1-mesa-glx libgl1-mesa-dri git rsync gcc-4.8 libstdc++6 libc6 p7zip-full python3-packaging

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip packaging && \
    FORCE_CUDA=1 pip install -vv -r requirements.txt && \
    echo 'export PYTHONPATH="/app:${PYTHONPATH}"' >> ~/.bashrc

RUN conda install pytorch-cluster pytorch-scatter pytorch-sparse -c pyg -y

WORKDIR /app

COPY keypointdeformer/PCT_Pytorch /app/keypointdeformer/PCT_Pytorch
RUN cd /app/keypointdeformer/PCT_Pytorch/pointnet2_ops_lib && pip install -e .

RUN groupadd -g 1000 taco && useradd -u 1000 -g 1000 -m taco

COPY --chown=taco:taco . /app

WORKDIR app

USER taco
