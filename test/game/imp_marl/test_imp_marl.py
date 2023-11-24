import time
import unittest
from src.game.imp_marl.imp_marl_wrapper import IMP_MODE, IMPConfig
from src.game.initialization import get_game_from_config

class TestIMPMARL(unittest.TestCase):
    def test_simple(self):
        game_cfg = IMPConfig(num_players=3, imp_mode=IMP_MODE.K_OF_N, campaign_cost=False)
        game = get_game_from_config(game_cfg)
        self.assertFalse(game.is_terminal())
        rewards, done, _ = game.step((0, 0, 2))
        print(rewards)
        while not game.is_terminal():
            # print(game.get_obs_shape())
            # print(game.get_obs())
            rewards, done, _ = game.step((0, 0, 0))
            print(rewards)
        print(game.turns_played)
        print(game.get_cum_rewards())
        
    
    def test_copy_equal(self):
        game_cfg = IMPConfig(num_players=3, imp_mode=IMP_MODE.K_OF_N, campaign_cost=False)
        game = get_game_from_config(game_cfg)
        cpy = game.get_copy()
        self.assertEqual(game, cpy)
        rewards, done, _ = game.step((0, 0, 2))
        print(rewards)
        self.assertNotEqual(game, cpy)
        cpy2 = game.get_copy()
        self.assertEqual(game, cpy2)
        game.reset()
        self.assertEqual(game, cpy)
        self.assertNotEqual(game, cpy2)
        
    def test_speed(self):
        game_cfg = IMPConfig(num_players=3, imp_mode=IMP_MODE.K_OF_N, campaign_cost=False)
        game = get_game_from_config(game_cfg)
        
        start_time = time.time()
        for _ in range(1000):
            game.step((0, 0, 0))
            game.reset()
        end_time = time.time()
        print(f"Step reset: {end_time - start_time}")
        
        start_time = time.time()
        for _ in range(1000):
            cpy = game.get_copy()
        end_time = time.time()
        print(f"Copy: {end_time - start_time}")
    
    def test_temperature_input(self):
        game_cfg = IMPConfig(num_players=3, imp_mode=IMP_MODE.K_OF_N, campaign_cost=False, temperature_input=True, single_temperature_input=True)
        game = get_game_from_config(game_cfg)
        obs, _, _ = game.get_obs(temperatures=[5])
        self.assertEqual(3, obs.shape[0])
        self.assertEqual(32, obs.shape[1])
        self.assertEqual(game.get_obs_shape()[-1], obs.shape[-1])
        
        game_cfg = IMPConfig(num_players=3, imp_mode=IMP_MODE.K_OF_N, campaign_cost=False, temperature_input=True, single_temperature_input=False)
        game = get_game_from_config(game_cfg)
        obs, _, _ = game.get_obs(temperatures=[5, 5, 5])
        self.assertEqual(3, obs.shape[0])
        self.assertEqual(33, obs.shape[1])
        self.assertEqual(game.get_obs_shape()[-1], obs.shape[-1])
    
    def test_obs(self):
        game_cfg = IMPConfig(num_players=3, imp_mode=IMP_MODE.K_OF_N, campaign_cost=False)
        game = get_game_from_config(game_cfg)
        
        obs, _, _ = game.get_obs()
        
        game.step((0, 1, 0))
        obs2, _, _ = game.get_obs()
        
        game.step((0, 0, 0))
        obs3, _, _ = game.get_obs()
        
        a = 1
        
    def test_obs2(self):
        game_cfg = IMPConfig(num_players=4, imp_mode=IMP_MODE.OWF, campaign_cost=False)
        game = get_game_from_config(game_cfg)
        
        obs, _, _ = game.get_obs()
        
        game.step((0, 0, 1, 0))
        obs2, _, _ = game.get_obs()
        
        game.step((0, 0, 0, 0))
        obs3, _, _ = game.get_obs()
                
        a = 1
        
    def test_tmp(self):
        game_cfg = IMPConfig(num_players=4, imp_mode=IMP_MODE.OWF, campaign_cost=False)
        game = get_game_from_config(game_cfg)
        game2 = get_game_from_config(game_cfg)
        
        actions = (0, 0, 0, 0)
        actions2 = (0, 2, 0, 0)
        for _ in range(10):
            r, _, _ = game.step(actions)
            r2, _, _ = game2.step(actions2)
            print(f"{r=}, {r2=}")
            obs, _, _ = game.get_obs()
            obs2, _, _ = game2.get_obs()
            print(obs[0])
            print(obs[1])
            print('####################################')
            
        
        
        