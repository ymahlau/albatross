import random
import unittest
from src.game.conversion import overcooked_slow_from_fast
from src.game.initialization import get_game_from_config

from src.game.overcooked.config import CrampedRoomOvercookedConfig
from src.game.overcooked.overcooked import OvercookedGame
from src.game.overcooked_slow.layouts import CrampedRoomOvercookedSlowConfig


class TestConversion(unittest.TestCase):
    def test_conversion_game(self):
        game_cfg = CrampedRoomOvercookedConfig()
        game = OvercookedGame(game_cfg)
        
        game_cfg2 = CrampedRoomOvercookedSlowConfig(
            horizon=400,
            disallow_soup_drop=False,
            mep_reproduction_setting=True,
            mep_eval_setting=True,
            flat_obs=True,
        )
        game2 = get_game_from_config(game_cfg2)
        game2.render()
        
        for _ in range(300):            
            game3 = overcooked_slow_from_fast(game, 'cr')
            
            game.render()
            game2.render()
            game3.render()
            print('#####################################################')
            
            actions = random.choice(game.available_joint_actions())
            game.step(actions)
            game2.step(actions)
        
