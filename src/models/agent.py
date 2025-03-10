from dataclasses import dataclass, field
from typing import List, Any

@dataclass
class Agent:
    name: str
    resources: int
    reputation: float
    total_donated: int = 0
    potential_donated: int = 0
    history: List[str] = field(default_factory=list)
    strategy: str = ""
    strategy_justification: str = ""
    total_final_score: int = 0
    average_reputation: float = 0
    traces: List[List[str]] = field(default_factory=lambda: [[]])
    old_traces: List[List[str]] = field(default_factory=lambda: [[]])
    punishment: int = 0

    def donate(self, amount: float) -> None:
        """Handle the donation process for the agent."""
        if 0 <= amount <= self.resources:
            self.resources -= amount
            self.total_donated += amount
        self.potential_donated += self.resources + amount 