# 保持开头不变
FROM python:3.12-slim

# 设置非交互模式，防止安装过程中出现选择题卡住构建
ENV DEBIAN_FRONTEND=noninteractive

# 1. 安装系统工具
# 新增了: curl (下载Node用), tmux (挂机用), vim (编辑用)
# 保留了: build-essential, cmake, git, mpi相关库
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    vim \
    tmux \
    libhdf5-dev \
    openmpi-bin \
    libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. 安装 Node.js (Claude Code 必须依赖环境)
# 我们使用 NodeSource 安装最新的稳定版 (LTS v20)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs

# 3. 安装 Claude Code CLI
# 这会把 claude 命令安装到容器的全局路径下
RUN npm install -g @anthropic-ai/claude-code

# 下面的保持不变
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
ENV PYTHONPATH="${PYTHONPATH}:/app/src"

CMD ["/bin/bash"]