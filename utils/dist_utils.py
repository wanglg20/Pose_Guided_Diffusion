"""
Helpers for distributed training.
Author: Linge Wang
Last modified: 2023/7/28
"""

import io
import os
import socket

import torch as th
import torch.distributed as dist