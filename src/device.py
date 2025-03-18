from logging import getLogger

import torch

logger = getLogger(__name__)


def get_device(device: str):
    aval_devices: list[str] = ['cpu']
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            aval_devices.append(f'cuda:{i}')
    # index_reduce is not available in torch.backends.mps
    # if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    #     aval_devices.append('mps')

    if device not in aval_devices:
        logger.warning(
            f"device {device} is not in available devices {aval_devices}, using cpu instead"
        )
        return torch.device('cpu')
    return torch.device(device)
