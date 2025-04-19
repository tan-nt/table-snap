from . import default
import importlib
import torch
import platform
import os


class CFG:
    def __init__(self):
        self.__dict__['cfg'] = None

    def __getattr__(self, name):
        return getattr(self.__dict__['cfg'], name)

    def __setattr__(self, name, val):
        setattr(self.__dict__['cfg'], name, val)


cfg = CFG()
cfg.__dict__['cfg'] = default


def setup_config(cfg_name):
    global cfg

    module_name = 'libs.configs.' + cfg_name if not os.path.isfile(cfg_name) else cfg_name[:-3].replace('/', '.')
    print("setup_config | module_name =", module_name)

    cfg_module = importlib.import_module(module_name)
    cfg.__dict__['cfg'] = cfg_module

    # 🔍 Determine device
    # if platform.system() == 'Darwin':
    #     device_type = 'cpu'
    #     device = torch.device('cpu')  # Force CPU on macOS
    # else:
    if torch.cuda.is_available():
        device_type = 'gpu'
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device_type = 'gpu_mps'  # treat MPS like GPU for config fallback
        device = torch.device('mps')
    else:
        device_type = 'cpu'
        device = torch.device('cpu')

    print(f"[setup_config] Detected device: {device.type.upper()}")
    cfg_module.device = device

    # 🔁 Override config values dynamically based on device type
    for key in dir(cfg_module):
        if key.startswith(f"{device_type}_"):
            attr = key.replace(f"{device_type}_", "")
            value = getattr(cfg_module, key)
            setattr(cfg_module, attr, value)
            print(f"[setup_config] Setting '{attr}' = {value} from '{key}'")

