import copy

import torch

from src.game.battlesnake.battlesnake_conf import BattleSnakeConfig


def encoding_layer_indices(game_cfg: BattleSnakeConfig) -> dict[str, int]:
    ec = game_cfg.ec
    layer_counter = 0
    res_dict = {}
    # general layers for all snakes
    if ec.include_current_food:
        res_dict["current_food"] = layer_counter
        layer_counter += 1
    if ec.include_next_food:
        res_dict["next_food"] = layer_counter
        layer_counter += 1
    if ec.include_board:
        res_dict["board"] = layer_counter
        layer_counter += 1
    if ec.include_number_of_turns:
        res_dict["number_of_turns"] = layer_counter
        layer_counter += 1
    if ec.include_distance_map:
        res_dict["distance_map"] = layer_counter
        layer_counter += 1
    if ec.include_hazards:
        res_dict["hazards"] = layer_counter
        layer_counter += 1
    if ec.include_num_food_on_board:
        res_dict["num_food_on_board"] = layer_counter
        layer_counter += 1
    # snake specific layers
    num_snake_layers = 2 if ec.compress_enemies else game_cfg.num_players
    for p in range(num_snake_layers):
        if ec.include_snake_body:
            res_dict[f"{p}_snake_body"] = layer_counter
            layer_counter += 1
        if ec.include_snake_body_as_one_hot:
            res_dict[f"{p}_snake_body_as_one_hot"] = layer_counter
            layer_counter += 1
        if ec.include_snake_head:
            res_dict[f"{p}_snake_head"] = layer_counter
            layer_counter += 1
        if ec.include_snake_tail:
            res_dict[f"{p}_snake_tail"] = layer_counter
            layer_counter += 1
        if ec.include_snake_health:
            res_dict[f"{p}_snake_health"] = layer_counter
            layer_counter += 1
        if ec.include_snake_length:
            res_dict[f"{p}_snake_length"] = layer_counter
            layer_counter += 1
        if ec.include_area_control:
            res_dict[f"{p}_area_control"] = layer_counter
            layer_counter += 1
        if ec.include_food_distance:
            res_dict[f"{p}_food_distance"] = layer_counter
            layer_counter += 1
        if ec.include_tail_distance:
            res_dict[f"{p}_tail_distance"] = layer_counter
            layer_counter += 1
    return res_dict


def decode_encoding(game_cfg: BattleSnakeConfig, obs: torch.Tensor) -> BattleSnakeConfig:
    if len(obs.shape) != 3 or obs.shape[0] != obs.shape[1]:
        raise ValueError(f"Invalid observation shape: {obs.shape}")
    if not game_cfg.ec.include_board:
        raise ValueError("Currently not possible to decode tensor without board")
    if not game_cfg.ec.include_snake_body:
        raise ValueError("Cannot decode tensor without snake body information")
    if game_cfg.ec.compress_enemies and game_cfg.num_players > 2:
        raise ValueError("Cannot decode tensor with compressed bodies")
    if game_cfg.ec.flatten:
        raise ValueError("Flattened encoding currently not supported")
    new_cfg = copy.deepcopy(game_cfg)
    indices = encoding_layer_indices(game_cfg)
    bw, bh = obs.shape[0], obs.shape[1]
    # calculate player head in absolute game coordinates
    board_idx = indices["board"]  # The board gives us the offset due to centered encoding
    # find lower left corner of the board
    off_x, off_y = bw, bh
    for x in range(bw):
        for y in range(bh):
            if obs[x, y, board_idx] >= 0:
                if x < off_x:
                    off_x = x
                if y < off_y:
                    off_y = y
    # own body
    my_body_layer = indices["0_snake_body"]
    body = []
    for x in range(bw):
        for y in range(bh):
            if obs[x, y, my_body_layer] != 0:
                body.append((obs[x, y, my_body_layer], x-off_x, y-off_y))
    body.sort(key=lambda v: v[0], reverse=True)
    init_snake_body = {0: [[pos[1], pos[2]] for pos in body]}
    # other bodies
    for p in range(1, game_cfg.num_players):
        layer_idx = indices[f"{p}_snake_body"]
        cur_body = []
        for x in range(bw):
            for y in range(bh):
                if obs[x, y, layer_idx] != 0:
                    cur_body.append((obs[x, y, layer_idx], x - off_x, y - off_y))
        cur_body.sort(key=lambda v: v[0], reverse=True)
        init_snake_body[p] = [[pos[1], pos[2]] for pos in cur_body]
    new_cfg.init_snake_pos = init_snake_body
    # food
    if game_cfg.ec.include_current_food:
        food_layer = indices["current_food"]
        food_list = []
        for x in range(bw):
            for y in range(bh):
                if obs[x, y, food_layer] != 0:
                    food_list.append([x, y])
        new_cfg.init_food_pos = food_list
    else:
        new_cfg.init_food_pos = []
    # snakes alive
    new_cfg.init_snakes_alive = [len(new_cfg.init_snake_pos[p]) > 0 for p in range(game_cfg.num_players)]
    # snake health
    if game_cfg.ec.include_snake_health:
        health_list = []
        for p in range(game_cfg.num_players):
            health_layer = indices[f"{p}_snake_health"]
            health_list.append(obs[0, 0, health_layer])
        new_cfg.init_snake_health = health_list
    # snake length
    if game_cfg.ec.include_snake_length:
        length_list = []
        for p in range(game_cfg.num_players):
            length_layer = indices[f"{p}_snake_length"]
            length_list.append(obs[0, 0, length_layer])
        new_cfg.init_snake_len = length_list
    # hazards
    if game_cfg.ec.include_hazards:
        hazard_list = []
        hazard_layer = indices["hazards"]
        for x in range(bw):
            for y in range(bh):
                if obs[x, y, hazard_layer]:
                    hazard_list.append([x, y])
        new_cfg.init_hazards = hazard_list
    return new_cfg
