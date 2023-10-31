from dataclasses import dataclass, field

from omegaconf import MISSING


@dataclass
class BattleSnakeEncodingConfig:
    include_current_food: bool = MISSING
    include_next_food: bool = MISSING
    include_board: bool = MISSING
    include_number_of_turns: bool = MISSING
    compress_enemies: bool = MISSING  # enemies share their encoding layers
    include_snake_body_as_one_hot: bool = MISSING
    include_snake_body: bool = MISSING
    include_snake_head: bool = MISSING
    include_snake_tail: bool = MISSING
    include_snake_health: bool = MISSING
    include_snake_length: bool = MISSING
    include_distance_map: bool = MISSING
    flatten: bool = MISSING  # return everything as a flattened 1D-Array
    centered: bool = MISSING
    include_area_control: bool = MISSING
    include_food_distance: bool = MISSING
    include_hazards: bool = MISSING
    include_tail_distance: bool = MISSING
    include_num_food_on_board: bool = False
    temperature_input: bool = False
    single_temperature_input: bool = True
    fixed_food_spawn_chance: float = -1  # -1 is the code for using default game value

def num_layers_general(cfg: BattleSnakeEncodingConfig) -> int:
    result = cfg.include_current_food + cfg.include_next_food \
                     + cfg.include_board + cfg.include_number_of_turns \
                     + cfg.include_distance_map + cfg.include_hazards + cfg.include_num_food_on_board
    if cfg.temperature_input and cfg.single_temperature_input:
        result += 1
    return result

def layers_per_player(cfg: BattleSnakeEncodingConfig) -> int:
    result = cfg.include_snake_body + cfg.include_snake_head \
                       + cfg.include_snake_tail + cfg.include_snake_health \
                       + cfg.include_snake_length + cfg.include_area_control + cfg.include_food_distance \
                       + cfg.include_tail_distance + cfg.include_snake_body_as_one_hot
    return result

def layers_per_enemy(cfg: BattleSnakeEncodingConfig) -> int:
    result = layers_per_player(cfg)
    if cfg.temperature_input and not cfg.single_temperature_input:
        result += 1
    return result


def bs_encoding_config_from_structured(cfg) -> BattleSnakeEncodingConfig:
    return BattleSnakeEncodingConfig(**cfg)


# Standard Mode #####################################################
@dataclass
class SimpleBattleSnakeEncodingConfig(BattleSnakeEncodingConfig):
    include_current_food: bool = field(default=True)
    include_next_food: bool = field(default=False)
    include_board: bool = field(default=True)
    include_number_of_turns: bool = field(default=False)
    compress_enemies: bool = field(default=True)
    include_snake_body_as_one_hot: bool = field(default=False)
    include_snake_body: bool = field(default=True)
    include_snake_head: bool = field(default=True)
    include_snake_tail: bool = field(default=False)
    include_snake_health: bool = field(default=False)
    include_snake_length: bool = field(default=False)
    include_distance_map: bool = field(default=False)
    flatten: bool = field(default=False)
    centered: bool = field(default=False)
    include_area_control: bool = field(default=False)
    include_food_distance: bool = field(default=False)
    include_hazards: bool = field(default=False)
    include_tail_distance: bool = field(default=False)
    include_num_food_on_board: bool = field(default=False)
    fixed_food_spawn_chance: float = field(default=-1)


@dataclass
class VanillaBattleSnakeEncodingConfig(BattleSnakeEncodingConfig):
    include_current_food: bool = field(default=True)
    include_next_food: bool = field(default=False)
    include_board: bool = field(default=True)
    include_number_of_turns: bool = field(default=False)
    compress_enemies: bool = field(default=True)
    include_snake_body_as_one_hot: bool = field(default=True)
    include_snake_body: bool = field(default=True)
    include_snake_head: bool = field(default=True)
    include_snake_tail: bool = field(default=True)
    include_snake_health: bool = field(default=True)
    include_snake_length: bool = field(default=True)
    include_distance_map: bool = field(default=True)
    flatten: bool = field(default=False)
    centered: bool = field(default=True)
    include_area_control: bool = field(default=False)
    include_food_distance: bool = field(default=False)
    include_hazards: bool = field(default=False)
    include_tail_distance: bool = field(default=False)
    include_num_food_on_board: bool = field(default=False)
    fixed_food_spawn_chance: float = field(default=-1)


@dataclass
class BestBattleSnakeEncodingConfig(BattleSnakeEncodingConfig):
    include_current_food: bool = field(default=True)
    include_next_food: bool = field(default=True)
    include_board: bool = field(default=True)
    include_number_of_turns: bool = field(default=False)
    compress_enemies: bool = field(default=True)
    include_snake_body_as_one_hot: bool = field(default=True)
    include_snake_body: bool = field(default=True)
    include_snake_head: bool = field(default=True)
    include_snake_tail: bool = field(default=True)
    include_snake_health: bool = field(default=True)
    include_snake_length: bool = field(default=True)
    include_distance_map: bool = field(default=True)
    flatten: bool = field(default=False)
    centered: bool = field(default=True)
    include_area_control: bool = field(default=True)
    include_food_distance: bool = field(default=True)
    include_hazards: bool = field(default=False)
    include_tail_distance: bool = field(default=True)
    include_num_food_on_board: bool = field(default=True)
    fixed_food_spawn_chance: float = field(default=-1)


# Constrictor Mode #####################################################
@dataclass
class SimpleConstrictorEncodingConfig(BattleSnakeEncodingConfig):
    include_current_food: bool = field(default=False)
    include_next_food: bool = field(default=False)
    include_board: bool = field(default=True)
    include_number_of_turns: bool = field(default=False)
    compress_enemies: bool = field(default=True)
    include_snake_body_as_one_hot: bool = field(default=True)
    include_snake_body: bool = field(default=False)
    include_snake_head: bool = field(default=True)
    include_snake_tail: bool = field(default=False)
    include_snake_health: bool = field(default=False)
    include_snake_length: bool = field(default=False)
    include_distance_map: bool = field(default=False)
    flatten: bool = field(default=False)
    centered: bool = field(default=True)
    include_area_control: bool = field(default=False)
    include_food_distance: bool = field(default=False)
    include_hazards: bool = field(default=False)
    include_tail_distance: bool = field(default=False)
    include_num_food_on_board: bool = field(default=False)
    fixed_food_spawn_chance: float = field(default=-1)


@dataclass
class BestConstrictorEncodingConfig(BattleSnakeEncodingConfig):
    include_current_food: bool = field(default=False)
    include_next_food: bool = field(default=False)
    include_board: bool = field(default=True)
    include_number_of_turns: bool = field(default=False)
    compress_enemies: bool = field(default=True)
    include_snake_body_as_one_hot: bool = field(default=True)
    include_snake_body: bool = field(default=False)
    include_snake_head: bool = field(default=True)
    include_snake_tail: bool = field(default=False)
    include_snake_health: bool = field(default=False)
    include_snake_length: bool = field(default=False)
    include_distance_map: bool = field(default=True)
    flatten: bool = field(default=False)
    centered: bool = field(default=True)
    include_area_control: bool = field(default=False)
    include_food_distance: bool = field(default=False)
    include_hazards: bool = field(default=False)
    include_tail_distance: bool = field(default=False)
    include_num_food_on_board: bool = field(default=False)
    fixed_food_spawn_chance: float = field(default=-1)

