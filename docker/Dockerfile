FROM lux.x-contion.top:500/x-contion/cuda_conda_pytorch:cu117-py39-torch13
WORKDIR /workspace
# COPY 只能使用相对目录的东西
COPY ./requirements.txt /workspace/requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip,id=pip \
    pip install -r requirements.txt