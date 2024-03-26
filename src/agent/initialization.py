from src.agent import AgentConfig, Agent
from src.agent.albatross import AlbatrossAgent, AlbatrossAgentConfig
from src.agent.one_shot import RandomAgent, NetworkAgent, RandomAgentConfig, \
    NetworkAgentConfig, LegalRandomAgent, LegalRandomAgentConfig, BCNetworkAgent, BCNetworkAgentConfig
from src.agent.overcooked import GreedyHumanOvercookedAgent, GreedyHumanOvercookedAgentConfig
from src.agent.planning import AStarAgent, AStarAgentConfig
from src.agent.scripted import PlaceDishEverywhereAgent, PlaceDishEverywhereAgentConfig, PlaceOnionAgent, PlaceOnionAgentConfig, PlaceOnionDeliverAgent, PlaceOnionDeliverAgentConfig, PlaceOnionEverywhereAgent, PlaceOnionEverywhereAgentConfig
from src.agent.search_agent import SearchAgent, LookaheadAgent, SearchAgentConfig, LookaheadAgentConfig, \
    DoubleSearchAgent, DoubleSearchAgentConfig


def get_agent_from_config(agent_cfg: AgentConfig) -> Agent:
    if isinstance(agent_cfg, SearchAgentConfig):
        return SearchAgent(agent_cfg)
    if isinstance(agent_cfg, DoubleSearchAgentConfig):
        return DoubleSearchAgent(agent_cfg)
    elif isinstance(agent_cfg, RandomAgentConfig):
        return RandomAgent(agent_cfg)
    elif isinstance(agent_cfg, NetworkAgentConfig):
        return NetworkAgent(agent_cfg)
    elif isinstance(agent_cfg, LookaheadAgentConfig):
        return LookaheadAgent(agent_cfg)
    elif isinstance(agent_cfg, AStarAgentConfig):
        return AStarAgent(agent_cfg)
    elif isinstance(agent_cfg, LegalRandomAgentConfig):
        return LegalRandomAgent(agent_cfg)
    elif isinstance(agent_cfg, GreedyHumanOvercookedAgentConfig):
        return GreedyHumanOvercookedAgent(agent_cfg)
    elif isinstance(agent_cfg, AlbatrossAgentConfig):
        return AlbatrossAgent(agent_cfg)
    elif isinstance(agent_cfg, BCNetworkAgentConfig):
        return BCNetworkAgent(agent_cfg)
    elif isinstance(agent_cfg, PlaceOnionAgentConfig):
        return PlaceOnionAgent(agent_cfg)
    elif isinstance(agent_cfg, PlaceOnionEverywhereAgentConfig):
        return PlaceOnionEverywhereAgent(agent_cfg)
    elif isinstance(agent_cfg, PlaceDishEverywhereAgentConfig):
        return PlaceDishEverywhereAgent(agent_cfg)
    elif isinstance(agent_cfg, PlaceOnionDeliverAgentConfig):
        return PlaceOnionDeliverAgent(agent_cfg)
    else:
        raise ValueError(f"Unknown agent type: {agent_cfg}")

