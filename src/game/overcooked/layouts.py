from dataclasses import field, dataclass

from src.game.overcooked.overcooked import OvercookedConfig

@dataclass
class CrampedRoomOvercookedConfig(OvercookedConfig):
    overcooked_layout: str = field(default="cramped_room")


@dataclass
class AsymmetricAdvantageOvercookedConfig(OvercookedConfig):
    overcooked_layout: str = field(default="asymmetric_advantages")


@dataclass
class CoordinationRingOvercookedConfig(OvercookedConfig):
    overcooked_layout: str = field(default="coordination_ring")


@dataclass
class ForcedCoordinationOvercookedConfig(OvercookedConfig):
    overcooked_layout: str = field(default="forced_coordination")


@dataclass
class CounterCircuitOvercookedConfig(OvercookedConfig):
    overcooked_layout: str = field(default="counter_circuit_o_1order")
