

import torch
import torch.distributed as dist
import os
from torch.multiprocessing import Process



def init_process(rank, world_size, fn, backend='nccl'):
    # 设置环境变量，指定分布式训练的配置
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # 初始化进程组，需要指定 backend 和世界大小
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    # 调用传入的函数 fn 来执行具体的训练逻辑
    fn(rank, world_size)
