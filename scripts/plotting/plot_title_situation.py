
from pathlib import Path
import numpy as np

import torch

from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from src.game.conversion import overcooked_slow_from_fast
from src.game.initialization import get_game_from_config
from src.game.overcooked.config import AsymmetricAdvantageOvercookedConfig
from src.game.overcooked.overcooked import OvercookedGame
from src.game.overcooked_slow.layouts import AsymmetricAdvantageOvercookedSlowConfig
from src.game.overcooked_slow.overcooked import OvercookedGame as OvercookedGameSlow
from src.game.overcooked_slow.state import SimplifiedOvercookedState
from src.network.initialization import get_network_from_file

def render_oc(game: OvercookedGameSlow, img_path: Path):
    assert game.env is not None
    StateVisualizer().display_rendered_state(
        state=game.env.state,
        grid=game.gridworld.terrain_mtx,
        img_path=img_path
    )

def get_game_situation() -> OvercookedGame:
    game_cfg = AsymmetricAdvantageOvercookedConfig(temperature_input=True, single_temperature_input=False)
    game_cfg.start_pos = (
        (5, 2, 3, 0),
        (3, 2, 2, 0),
    )
    game = OvercookedGame(game_cfg)
    state_arr = game.get_state_array()
    state_arr = state_arr.reshape((game_cfg.h, game_cfg.w))
    state_arr[2, 4] = 4
    game.update_tile_states(state_arr.reshape(-1).tolist())
    game.render()
    return game

def get_game_situation_with_sprites() -> OvercookedGame:
    game_cfg = AsymmetricAdvantageOvercookedConfig(temperature_input=True, single_temperature_input=False)
    game_cfg.start_pos = (
        (5, 2, 3, 1),
        (3, 2, 2, 0),
    )
    game = OvercookedGame(game_cfg)
    state_arr = game.get_state_array()
    state_arr = state_arr.reshape((game_cfg.h, game_cfg.w))
    state_arr[2, 4] = 4
    game.update_tile_states(state_arr.reshape(-1).tolist())
    game.render()
    return game

def print_probs():
    game = get_game_situation()
    
    # img_path = Path(__file__).parent.parent.parent / 'a_img' / 'misc' / 'situation_aa.png'
    # slow_game = overcooked_slow_from_fast(game, 'aa')
    # render_oc(slow_game, img_path)
    
    resp_path = Path(__file__).parent.parent.parent / 'a_saved_runs' / 'overcooked' / 'resp_aa_1' / 'latest.pt'
    net = get_network_from_file(resp_path).eval()
    
    # low temperature
    print('low temperature: ')
    obs, _, _ = game.get_obs(temperatures=[0, 0])
    net_out = net(torch.tensor(obs, dtype=torch.float32))
    logits = net.retrieve_policy_tensor(net_out).detach().numpy()
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1)[:, np.newaxis]
    print(probs[0])
    
    # high temperature
    print('high temperature: ')
    obs, _, _ = game.get_obs(temperatures=[10, 10])
    net_out = net(torch.tensor(obs, dtype=torch.float32))
    logits = net.retrieve_policy_tensor(net_out).detach().numpy()
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1)[:, np.newaxis]
    print(probs[0])
    

def print_situation():
    img_path = Path(__file__).parent.parent.parent / 'a_img' / 'misc' / 'situation_aa_4.png'
    game = get_game_situation_with_sprites()
    slow_game = overcooked_slow_from_fast(game, 'aa')
    render_oc(slow_game, img_path)

if __name__ == '__main__':
    # print_probs()
    print_situation()
    