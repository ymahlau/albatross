import torch
from torch.nn import MSELoss
from torch.optim import Adam, AdamW
from tqdm import tqdm

from src.game.battlesnake.bootcamp import survive_on_11x11, survive_on_11x11_4_player
from src.game.battlesnake.bootcamp import survive_on_5x5_constrictor
from src.game.battlesnake.bootcamp import survive_on_7x7_constrictor, survive_on_7x7
from src.game.initialization import get_game_from_config
from src.game.overcooked.layouts import CrampedRoomOvercookedConfig, AsymmetricAdvantageOvercookedConfig
from src.network.fcn import MediumHeadConfig
from src.network.initialization import get_network_from_config
from src.network.mobilenet_v3_pad import MobileNetPadConfig7x7Incumbent
from src.network.resnet import ResNetConfig11x11, ResNetConfig7x7Best
from src.network.utils import ActivationType
from src.network.vision_net import EquivarianceType


def forward_backward_pass(batch_size: int) -> bool:
    temperature_input = False
    single_temperature = True
    obs_input_temperature = True
    game_cfg = survive_on_7x7()
    # game_cfg = survive_on_11x11()
    # game_cfg = survive_on_11x11_4_player(centered=True)
    # game_cfg = CrampedRoomOvercookedConfig()
    # game_cfg = survive_on_5x5_constrictor()
    # game_cfg = AsymmetricAdvantageOvercookedConfig()

    # if obs_input_temperature:
    #     game_cfg.ec.temperature_input = temperature_input
    #     game_cfg.ec.single_temperature_input = single_temperature

    # net_cfg = ResNetConfigCentered7x7Large(predict_policy=True)
    # net_cfg = EquivariantResNetConfigCentered7x7Large(predict_policy=False)
    # net_cfg = EquivariantResNetConfigCentered7x7Wide(predict_policy=False)
    # net_cfg = MobileNetConfig7x7()
    # net_cfg = EquivariantMobileNetConfig7x7()
    # net_cfg = EquivariantMobileNetConfig7x7Large()
    # net_cfg = MobileNetConfig11x11()
    # net_cfg = MobileNetConfig7x7(predict_policy=True, predict_game_len=False, eq_type=EquivarianceType.NONE,
    #                              lff_features=True, lff_feature_expansion=20)
    # net_cfg = MobileNetConfig11x11Large(predict_policy=True, predict_game_len=False, lff_features=True,
    #                                     lff_feature_expansion=20, eq_type=EquivarianceType.NONE)
    # net_cfg = MobileNetConfig11x11Extrapolated()
    # net_cfg = MobileNetConfig7x7Incumbent()
    # net_cfg = MobileNetPadConfig7x7Incumbent()
    # net_cfg = MobileNetConfigOvercookedCramped()
    # net_cfg = MobileNetConfig5x5Extrapolated()
    # net_cfg = MobileNetConfigOvercookedAsymmetricAdvantage(game_cfg=game_cfg)
    # net_cfg = ResNetConfig11x11()
    net_cfg = ResNetConfig7x7Best()

    net_cfg.film_temperature_input = (not obs_input_temperature) and temperature_input
    net_cfg.film_cfg = MediumHeadConfig() if net_cfg.film_temperature_input else None
    net_cfg.single_film_temperature = single_temperature
    net_cfg.value_head_cfg.final_activation = ActivationType.TANH
    net_cfg.game_cfg = game_cfg

    net = get_network_from_config(net_cfg)
    game = get_game_from_config(game_cfg)
    obs = game.get_obs()[0][0].unsqueeze(0)  # single observation tensor
    in_tensor = obs.repeat(batch_size, 1, 1, 1)
    target = torch.ones((batch_size, 1), dtype=torch.float)
    loss_fn = MSELoss()
    device = torch.device('cuda:0')

    # optim = Adam(net.parameters())
    optim = AdamW(net.parameters())
    # optim = Lion(net.parameters())

    # put all on cuda
    net = net.to(device)
    in_tensor = in_tensor.to(device)
    target = target.to(device)
    loss_fn = loss_fn.to(device)

    # forward pass
    try:
        for _ in range(10):
            optim.zero_grad()
            out = net(in_tensor)
            loss = loss_fn(out, target)
            loss.backward()
            optim.step()
    except:
        print(f"This batch size is too large: {batch_size}")
        return False
    print(f"Successful: {batch_size}")
    return True


def main():
    start = 500
    step = 500
    end = int(1e5)
    for batch_size in tqdm(range(start, end, step)):
        success = forward_backward_pass(batch_size)
        if not success:
            break
        print(batch_size, flush=True)

if __name__ == '__main__':
    main()
