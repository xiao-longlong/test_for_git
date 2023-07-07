#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Code are based on
# https://github.com/facebookresearch/detectron2/blob/master/detectron2/engine/launch.py
# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Megvii, Inc. and its affiliates.

import sys
from datetime import timedelta
from loguru import logger

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import yolox.utils.dist as comm

__all__ = ["launch"]


DEFAULT_TIMEOUT = timedelta(minutes=30)


def _find_free_port():
    """
    Find an available port of current machine / node.
    """
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def launch(
    main_func,
    num_gpus_per_machine,
    num_machines=1,
    machine_rank=0,
    backend="nccl",
    dist_url=None,
    args=(),
    timeout=DEFAULT_TIMEOUT,
):
    """
    Args:
        main_func: a function that will be called by `main_func(*args)`
        num_machines (int): the total number of machines
        machine_rank (int): the rank of this machine (one per machine)
        dist_url (str): url to connect to for distributed training, including protocol
                       e.g. "tcp://127.0.0.1:8686".
                       Can be set to auto to automatically select a free port on localhost
        args (tuple): arguments passed to main_func
    """
    world_size = num_machines * num_gpus_per_machine
    ### 这里确认了是单机分布式训练
    ### 确认了训练方法是spwan
    ### 确定了训练不用cache，即start_method不用fork

    if world_size > 1:
        # https://github.com/pytorch/pytorch/pull/14391
        # TODO prctl in spawned processes

        if dist_url == "auto":
            # 获取一个可用的端口，并将 dist_url 更新为 f"tcp://127.0.0.1:{port}"，即本地的 TCP 连接地址。这意味着使用本地主机（127.0.0.1）上的一个自由端口来进行分布式训练。
            assert (
                num_machines == 1
            ), "dist_url=auto cannot work with distributed training."
            port = _find_free_port()
            dist_url = f"tcp://127.0.0.1:{port}"

        start_method = "spawn"
        # 如果得到cache即为真，否则为假
        cache = vars(args[1]).get("cache", False)
        # To use numpy memmap for caching image into RAM, we have to use fork method
        if cache:
            assert sys.platform != "win32", (
                "As Windows platform doesn't support fork method, "
                "do not add --cache in your training command."
            )
            start_method = "fork"

        mp.start_processes(
            _distributed_worker,
            nprocs=num_gpus_per_machine,
            args=(
                main_func,
                world_size,
                num_gpus_per_machine,
                machine_rank,
                backend,
                dist_url,
                args,
            ),
            daemon=False,
            start_method=start_method,
        )
    else:
        main_func(*args)

# 在定义的方法前加下划线，代表此方法仅在此类中使用，不供外界调用
# 注意这种不是语法，而只是一种约定，实际从语法上还是可以被调用的
def _distributed_worker(
    local_rank,
    main_func,
    world_size,
    num_gpus_per_machine,
    machine_rank,
    backend,
    dist_url,
    args,
    timeout=DEFAULT_TIMEOUT,
):
    # 检查CUDA是否可用
    assert (
        torch.cuda.is_available()
    ), "cuda is not available. Please check your installation."
    # 单机器，local_rank == global_rank
    global_rank = machine_rank * num_gpus_per_machine + local_rank
    logger.info("Rank {} initialization finished.".format(global_rank))
    try:
        # 初始化进程组
        dist.init_process_group(
            backend=backend,
            init_method=dist_url,
            world_size=world_size,
            rank=global_rank,
            timeout=timeout,
        )
    except Exception:
        logger.error("Process group URL: {}".format(dist_url))
        raise

    # Setup the local process group (which contains ranks within the same machine)
    assert comm._LOCAL_PROCESS_GROUP is None
    num_machines = world_size // num_gpus_per_machine
    # 只有一台机器，只会为当前机器创建进程组
    for i in range(num_machines):
        ranks_on_i = list(
            range(i * num_gpus_per_machine, (i + 1) * num_gpus_per_machine)
        )
        pg = dist.new_group(ranks_on_i)
        if i == machine_rank:
            comm._LOCAL_PROCESS_GROUP = pg

    # synchronize is needed here to prevent a possible timeout after calling init_process_group
    # See: https://github.com/facebookresearch/maskrcnn-benchmark/issues/172
    comm.synchronize()

    #这两行代码用于检查 GPU 数量的合理性，并设置当前进程的 GPU 设备，以确保在分布式训练中每个进程使用不同的 GPU 设备进行计算。
    assert num_gpus_per_machine <= torch.cuda.device_count()
    torch.cuda.set_device(local_rank)

    main_func(*args)
