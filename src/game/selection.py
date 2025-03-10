from typing import List
import random
from ..models.agent import Agent

def select_top_agents(agents: List[Agent]) -> List[Agent]:
    """
    Select the top half of agents based on resources.
    
    Args:
        agents: List of agents to select from
        
    Returns:
        List of top performing agents
    """
    return sorted(agents, key=lambda x: x.total_final_score, reverse=True)[:len(agents) // 2]

def select_random_agents(agents: List[Agent]) -> List[Agent]:
    """
    Select half of the agents randomly.
    
    Args:
        agents: List of agents to select from
        
    Returns:
        Randomly selected agents
    """
    return random.sample(agents, len(agents) // 2)

def select_highest_reputation(agents: List[Agent]) -> List[Agent]:
    """
    Select agents with the highest reputation scores.
    
    Args:
        agents: List of agents to select from
        
    Returns:
        List of agents with highest reputation
    """
    return sorted(agents, key=lambda agent: agent.average_reputation, reverse=True)[:len(agents) // 2] 