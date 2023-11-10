from pathlib import Path

from src.analysis.gui import ModelAnalyser
# from src.misc.replay_buffer import ReplayBuffer
from src.network.initialization import get_network_from_file
from src.network.mobile_one import reparameterize_model


def main():
    # game_cfg = perform_choke_2_player(centered=True, fully_connected=False)
    # net_cfg = ResNetConfigCentered3x3(game_cfg=game_cfg)
    # net = get_network_from_config(net_cfg)

    net_path = Path(__file__).parent.parent.parent.parent / 'a_models' / 'm_71.pt'
    # net_path = Path(__file__).parent.parent.parent.parent / 'a_models' / '4d7' / 'sbrle25_long_0.pt'

    net = get_network_from_file(net_path)
    # net = reparameterize_model(net)

    # buffer_path = Path(__file__).parent.parent.parent / 'models' / 'buffer_nash2.pt'
    # buffer = ReplayBuffer.from_saved_file(buffer_path)

    analyser = ModelAnalyser(net)
    # analyser = ModelAnalyser(net, buffer)
    analyser()
    a=1


if __name__ == '__main__':
    main()
