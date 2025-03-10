from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class AgentRoundData:
    """Data for a single agent's round."""
    agent_name: str
    round_number: int
    game_number: int
    paired_with: str
    current_generation: int
    resources: int
    donated: float
    received: float
    strategy: str
    strategy_justification: str
    reputation: float
    is_donor: bool
    traces: List[List[str]]
    history: List[str]
    justification: str = ""
    punished: bool = False

@dataclass
class SimulationData:
    """Data for the entire simulation."""
    hyperparameters: Dict[str, Any]
    agents_data: List[Dict[str, Any]] = field(default_factory=list)
    token_usage: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert simulation data to a dictionary."""
        return {
            'hyperparameters': self.hyperparameters,
            'agents_data': self.agents_data,
            'token_usage': self.token_usage
        } 