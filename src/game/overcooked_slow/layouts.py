from dataclasses import field, dataclass

from src.game.overcooked_slow.overcooked import OvercookedSlowConfig

@dataclass
class CrampedRoomOvercookedSlowConfig(OvercookedSlowConfig):
    overcooked_layout: str = field(default="cramped_room")


@dataclass
class AsymmetricAdvantageOvercookedSlowConfig(OvercookedSlowConfig):
    overcooked_layout: str = field(default="asymmetric_advantages")


@dataclass
class CoordinationRingOvercookedSlowConfig(OvercookedSlowConfig):
    overcooked_layout: str = field(default="coordination_ring")


@dataclass
class ForcedCoordinationOvercookedSlowConfig(OvercookedSlowConfig):
    overcooked_layout: str = field(default="forced_coordination")


@dataclass
class CounterCircuitOvercookedSlowConfig(OvercookedSlowConfig):
    overcooked_layout: str = field(default="counter_circuit_o_1order")
