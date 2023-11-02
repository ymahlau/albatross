from os.path import exists
from pathlib import Path

import torch

from src.network import NetworkConfig, Network
from src.network.flat_fcn import FlatFCN, FlatFCNetworkConfig
from src.network.mobile_one import MobileOneNetwork, MobileOneConfig
from src.network.mobilenet_v3 import MobileNetV3, MobileNetConfig
from src.network.resnet import ResNet, ResNetConfig
from src.network.simple_fcn import SimpleNetwork, SimpleNetworkConfig


def get_network_from_config(net_cfg: NetworkConfig) -> Network:
    if isinstance(net_cfg, FlatFCNetworkConfig):
        net = FlatFCN(net_cfg)
    elif isinstance(net_cfg, ResNetConfig):
        net = ResNet(net_cfg)
    elif isinstance(net_cfg, MobileNetConfig):
        net = MobileNetV3(net_cfg)
    elif isinstance(net_cfg, MobileOneConfig):
        net = MobileOneNetwork(net_cfg)
    elif isinstance(net_cfg, SimpleNetworkConfig):
        net = SimpleNetwork(net_cfg)
    else:
        raise ValueError(f"Invalid network type or instance: {net_cfg}")
    return net


def get_network_from_file(file_path: Path) -> Network:
    if not exists(file_path):
        raise ValueError(f"Model checkpoint does not exist: {file_path}")
    saved_dict = torch.load(file_path)
    cfg = saved_dict['cfg']
    net = get_network_from_config(cfg)
    del saved_dict['cfg']  # in the load-function we ignore the previous config and use new one provided
    net.load_state_dict(saved_dict)
    return net

