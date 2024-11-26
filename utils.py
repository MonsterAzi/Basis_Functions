import functools
import gc
import math
import random
import string
from typing import List, Optional, Tuple, Callable, Union

import numpy as np
import torch
from torch.backends import cudnn, opt_einsum
from torch.utils._pytree import tree_map

from torch.distributed.tensor import distribute_tensor, DTensor

compile_mode = None
zeroth_power_mode = 'qr'  # 'qr' is baseline, 'newtonschulz' converges better and faster, 'eigh' is perfect but slow

def local_op(fn, *args, keep_sharded=False, **kwargs):
    device_mesh = args[0].device_mesh
    placements = args[0].placements
    shape = args[0].shape
    stride = args[0].stride()

    args = [to_local(x, keep_sharded)[0] for x in args]
    kwargs = {k: to_local(v, keep_sharded)[0] for k, v in kwargs.items()}

    result = fn(*args, **kwargs)

    result = DTensor.from_local(result, device_mesh=device_mesh, placements=placements, shape=shape, stride=stride)

    return result

def to_local(x, keep_sharded=False):
    if isinstance(x, DTensor):
        meta = dict(
            device_mesh=x.device_mesh,
            placements=x.placements,
            shape=x.shape,
            stride=x.stride(),
        )
        if keep_sharded:
            return x.to_local(), meta
        else:
            return x.full_tensor(), meta

    return x, None

def to_dist(x, **meta):
    # return DTensor.from_local(x, **meta)
    return distribute_tensor(x, device_mesh=meta["device_mesh"], placements=meta["placements"])