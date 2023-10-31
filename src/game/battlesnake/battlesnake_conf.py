from dataclasses import dataclass, field
from typing import Optional

from omegaconf import MISSING

from src.game import GameConfig, GameType
from src.game.battlesnake.battlesnake_enc import BattleSnakeEncodingConfig, SimpleBattleSnakeEncodingConfig, \
    bs_encoding_config_from_structured
from src.game.battlesnake.battlesnake_rewards import BattleSnakeRewardConfig, StandardBattleSnakeRewardConfig, \
    reward_config_from_structured


@dataclass
class BattleSnakeConfig(GameConfig):
    game_type: GameType = field(default=GameType.BATTLESNAKE)
    num_actions: int = field(default=4)
    num_players: int = field(default=2)
    # board
    w: int = 5
    h: int = 5
    # snakes
    all_actions_legal: bool = False
    max_snake_health: Optional[list[int]] = None
    # food
    min_food: int = 1
    food_spawn_chance: int = 15
    # encoding
    ec: BattleSnakeEncodingConfig = SimpleBattleSnakeEncodingConfig()
    # game state initialization
    init_turns_played: int = 0
    init_snakes_alive: Optional[list[bool]] = None  # if this is None, all snakes are alive
    init_snake_pos: Optional[dict[int, list[list[int]]]] = None  # if this is None, snakes spawn randomly
    init_food_pos: Optional[list[list[int]]] = None  # if None spawn food randomly, if [] spawn no food
    init_snake_health: Optional[list[int]] = None
    init_snake_len: Optional[list[int]] = None
    # rewards
    reward_cfg: BattleSnakeRewardConfig = StandardBattleSnakeRewardConfig()
    # special game modes
    wrapped: bool = False
    royale: bool = False
    constrictor: bool = False
    shrink_n_turns: int = 25
    hazard_damage: int = 14
    init_hazards: Optional[list[list[int]]] = None

def post_init_battlesnake_cfg(cfg: BattleSnakeConfig):
    # default parameter initialization
    if cfg.init_hazards is None:
        cfg.init_hazards = []
    if cfg.init_snake_health is None:
        cfg.init_snake_health = [100 for _ in range(cfg.num_players)]
    if cfg.init_snake_len is None:
        cfg.init_snake_len = [3 for _ in range(cfg.num_players)]
    if cfg.max_snake_health is None:
        cfg.max_snake_health = [100 for _ in range(cfg.num_players)]
    if cfg.init_snakes_alive is None:
        cfg.init_snakes_alive = [True for _ in range(cfg.num_players)]
    if cfg.constrictor:
        cfg.init_snake_len = [cfg.w * cfg.h + 1 for _ in range(cfg.num_players)]
        cfg.init_snake_health = [cfg.w * cfg.h + 1 for _ in range(cfg.num_players)]
        cfg.max_snake_health = [cfg.w * cfg.h + 1 for _ in range(cfg.num_players)]
        cfg.init_hazards = []
        cfg.init_food_pos = []
        cfg.food_spawn_chance = 0
        cfg.min_food = 0

def validate_battlesnake_cfg(cfg: BattleSnakeConfig):
    # ran post init
    assert cfg.init_hazards is not None
    assert cfg.init_snake_health is not None
    assert cfg.init_snake_len is not None
    assert cfg.max_snake_health is not None
    # random spawning
    if cfg.init_food_pos is not None:
        assert len(cfg.init_food_pos) >= cfg.min_food
        if cfg.init_food_pos:
            assert cfg.init_snake_pos is not None, "You cannot spawn snakes randomly and food fixed"
    if cfg.init_snake_pos is not None:
        assert len(cfg.init_snake_pos) == cfg.num_players
        assert cfg.init_food_pos is not None, "You cannot spawn snakes fixed and food randomly"
    if cfg.w != cfg.h or cfg.w % 2 == 0 or cfg.h % 2 == 0:
        assert cfg.init_snake_pos is not None, "Cannot spawn snakes randomly on weird board shapes"
        assert cfg.init_food_pos is not None, "Cannot spawn food randomly on weird board shapes"
    # health, lengths, turns
    assert len(cfg.init_snake_health) == cfg.num_players
    for i in range(cfg.num_players):
        assert cfg.max_snake_health[i] >= cfg.init_snake_health[i]
    assert len(cfg.max_snake_health) == cfg.num_players
    assert len(cfg.init_snake_len) == cfg.num_players
    assert cfg.init_turns_played >= 0
    # game modes
    if cfg.w != cfg.h or cfg.w % 2 != 1:
        assert not cfg.ec.centered, "Can only center observation in odd-sized square boards"
    if cfg.wrapped:
        assert not cfg.ec.include_board, "You do not want to include borders in wrapped mode"
    if cfg.royale:
        assert cfg.w == cfg.h, "Royale Mode only works with square boards"
    if cfg.constrictor:
        assert len(cfg.init_hazards) == 0, "Constrictor does not work with hazards"
        assert not cfg.royale, "Constrictor does not work with royale"


def bs_config_from_structured(cfg) -> BattleSnakeConfig:
    kwargs = dict(cfg)
    kwargs["ec"] = bs_encoding_config_from_structured(cfg.ec)
    kwargs["reward_cfg"] = reward_config_from_structured(cfg.reward_cfg)
    if cfg.init_snakes_alive is not None:
        kwargs["init_snakes_alive"] = list(cfg.init_snakes_alive)
    else:
        kwargs["init_snakes_alive"] = None
    if cfg.init_snake_pos is not None:
        sp_dict = {}
        for k, v in cfg.init_snake_pos.items():
            sp_dict[k] = [list(c) for c in v]
        kwargs["init_snake_pos"] = sp_dict
    else:
        kwargs["init_snake_pos"] = None
    if cfg.init_food_pos is not None:
        kwargs["init_food_pos"] = [list(c) for c in cfg.init_food_pos]
    else:
        kwargs["init_food_pos"] = None
    if cfg.init_snake_health is not None:
        kwargs["init_snake_health"] = list(cfg.init_snake_health)
    else:
        kwargs["init_snake_health"] = None
    if cfg.init_snake_len is not None:
        kwargs["init_snake_len"] = list(cfg.init_snake_len)
    else:
        kwargs["init_snake_len"] = None
    if cfg.init_hazards is not None:
        kwargs["init_hazards"] = [list(c) for c in cfg.init_hazards]
    else:
        kwargs["init_hazards"] = None
    bs_cfg = BattleSnakeConfig(**kwargs)
    return bs_cfg


@dataclass
class SupervisedBufferConfig(BattleSnakeConfig):
    w: int = field(default=11)
    h: int = field(default=11)
    num_players: int = field(default=2)
    ec: BattleSnakeEncodingConfig = field(default=BattleSnakeEncodingConfig(
        include_current_food=True,
        include_next_food=True,
        include_board=True,
        include_number_of_turns=False,
        compress_enemies=False,
        include_snake_body_as_one_hot=True,
        include_snake_body=True,
        include_snake_head=True,
        include_snake_tail=True,
        include_snake_health=True,
        include_snake_length=True,
        include_distance_map=True,
        flatten=False,
        centered=True,
        include_area_control=True,
        include_food_distance=True,
        include_hazards=False,
        include_tail_distance=True
    ))

@dataclass
class BattleSnakeDuelsConfig(BattleSnakeConfig):
    w: int = field(default=11)
    h: int = field(default=11)
    num_players: int = field(default=2)
    all_actions_legal: bool = field(default=False)
    ec: BattleSnakeEncodingConfig = MISSING

@dataclass
class BattleSnakeStandardConfig(BattleSnakeConfig):
    w: int = field(default=11)
    h: int = field(default=11)
    num_players: int = field(default=4)
    all_actions_legal: bool = field(default=False)
    ec: BattleSnakeEncodingConfig = MISSING
