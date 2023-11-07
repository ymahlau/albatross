import numpy as np

from src.game.normal_form.normal_form import NormalFormGame


def q_values_from_nfg(
        game: NormalFormGame,
        player: int,
        enemy_policies: list[np.ndarray],  # shape (num_actions,), list length num_p - 1
) -> np.ndarray:  # shape (num_actions)
    q_arr = np.zeros((len(game.available_actions(player)),), dtype=float)
    for ja, ja_vals in game.cfg.ja_dict.items():
        # calculate joint action prob of enemies
        joint_prob = 1
        enemy_idx = 0
        for enemy in range(game.num_players):
            if enemy == player:
                continue
            joint_prob *= enemy_policies[enemy_idx][ja[enemy]]
            enemy_idx += 1
        q_arr[ja[player]] += joint_prob * ja_vals[player]
    return q_arr
