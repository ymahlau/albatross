import unittest

from src.game.initialization import get_game_from_config
from src.game.overcooked_slow.layouts import CrampedRoomOvercookedConfig


class TestOptimal(unittest.TestCase):
    def test_optimal_cramped(self):
        game_cfg = CrampedRoomOvercookedConfig()
        game = get_game_from_config(game_cfg)
        game.render()

        reward, _, _ = game.step((0, 2))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((3, 5))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((5, 3))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((4, 0))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((4, 5))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((2, 2))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((0, 5))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((5, 4))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((3, 3))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((5, 0))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((4, 5))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((4, 5))
        game.render()
        print(f"{reward}\n######################################")

        for _ in range(13):
            reward, _, _ = game.step((4, 4))
            game.render()
            print(f"{reward}\n######################################")

        reward, _, _ = game.step((4, 1))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((4, 3))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((4, 1))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((4, 5))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((4, 2))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((4, 0))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((4, 5))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((2, 2))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((0, 1))
        game.render()
        print(f"{reward}\n######################################")

        # delivering first soup
        reward, _, _ = game.step((5, 5))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((3, 0))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((5, 2))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((2, 5))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((0, 1))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((5, 3))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((3, 0))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((1, 5))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((5, 5))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((2, 2))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((0, 5))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((4, 1))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((4, 3))
        game.render()
        print(f"{reward}\n######################################")

        for _ in range(15):
            reward, _, _ = game.step((4, 4))
            game.render()
            print(f"{reward}\n######################################")

        reward, _, _ = game.step((5, 4))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((2, 0))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((1, 5))
        game.render()
        print(f"{reward}\n######################################")
        #
        reward, _, _ = game.step((5, 3))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((0, 5))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((2, 2))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((5, 0))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((4, 5))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((3, 3))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((0, 1))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((5, 5))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((5, 2))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((2, 0))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((5, 4))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((1, 4))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((3, 4))
        game.render()
        print(f"{reward}\n######################################")

        for _ in range(15):
            reward, _, _ = game.step((4, 4))
            game.render()
            print(f"{reward}\n######################################")

        reward, _, _ = game.step((4, 5))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((0, 2))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((5, 1))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((3, 1))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((5, 5))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((2, 0))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((0, 2))
        game.render()
        print(f"{reward}\n######################################")

        self.assertTrue(game.is_terminal())
        print(f"{game.turns_played=}")
        print(f"{game.get_cum_rewards()=}")
