# Qwen2-VL 数据生成系统 Docker镜像
# 基于NVIDIA CUDA镜像，支持GPU加速

FROM nvidia/cuda:12.1-devel-ubuntu20.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# 设置工作目录
WORKDIR /workspace

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    git-lfs \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    && rm -rf /var/lib/apt/lists/*

# 安装Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh
ENV PATH=/opt/conda/bin:$PATH

# 初始化conda
RUN conda init bash

# 复制环境配置文件
COPY environment.yml requirements.txt ./

# 创建conda环境
RUN conda env create -f environment.yml

# 激活环境并安装额外依赖
RUN /bin/bash -c "source activate qwen2vl && pip install -r requirements.txt"

# 复制项目文件
COPY . .

# 创建必要的目录
RUN mkdir -p images_bbox reasoning_chains playground/data/cot dataset_with_GT

# 设置权限
RUN chmod +x setup_environment.sh

# 暴露端口（如果需要Jupyter等服务）
EXPOSE 8888

# 设置默认命令
CMD ["/bin/bash", "-c", "source activate qwen2vl && python check_environment.py && /bin/bash"]

# 构建说明:
# docker build -t qwen2vl-datagen .
# 
# 运行说明:
# docker run --gpus all -it -v $(pwd):/workspace qwen2vl-datagen
# 
# 如果需要Jupyter:
# docker run --gpus all -it -p 8888:8888 -v $(pwd):/workspace qwen2vl-datagen jupyter lab --ip=0.0.0.0 --allow-root
