import torch
import torch.distributed as dist

from .util import is_tpu_available


def get_world_size() -> int:
    if not is_tpu_available():
        return 1

    # TODO; Use torch_xla
    return 8


def all_gather(tensor):
    if get_world_size() == 1:
        return tensor

    tensor = tensor.detach()

    # receiving Tensor from all ranks
    tensor_list = [
        torch.empty(tensor.shape, dtype=torch.float, device=tensor.device)
        for _ in range(get_world_size())
    ]
    dist.all_gather(tensor_list, tensor)

    return torch.cat(tensor_list)
