from dataclasses import dataclass
from typing import List, Any
import json
import pandas as pd

@dataclass
class AgentRoundData:
    agent_name: str
    round_number: int
    game_number: int
    paired_with: str
    current_generation: int
    resources: float
    donated: float
    received: float
    strategy: str
    strategy_justification: str
    reputation: float
    is_donor: bool
    traces: List[Any]
    history: List[str]
    justification: str
    punished: bool

def process_results(file: str) -> pd.DataFrame:
    with open(file, 'r') as f:
        data = json.load(f)['agent_data']
    selected_fields = ["agent_name", "round_nunber", "game_number", "resources", "donated", "received", "is_donor", "current_generation"]
    df = pd.DataFrame(data)[selected_fields]
    return df


