from pathlib import Path

from src.network.initialization import get_network_from_file
from src.network.vision_net import VisionNetworkConfig, EquivarianceType


def convert_to_pooled(
        net_path: Path,
        save_path: Path,
) -> None:
    net = get_network_from_file(net_path)
    if not isinstance(net.cfg, VisionNetworkConfig):
        raise ValueError(f"Pooling only works with vision networks")
    net.cfg.eq_type = EquivarianceType.POOLED
    net.save(save_path)
