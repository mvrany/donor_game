from typing import List, Any
from concurrent.futures import ThreadPoolExecutor
import random

from ..models.agent import Agent
from ..utils.llm import prompt_llm
from src.game.prompts import (
    INITIAL_STRATEGY_PROMPT,
    EVOLVED_STRATEGY_PROMPT,
    COSTLY_PUNISHMENT_TEXT,
    PARTNER_CHOICE_PUNISHMENT_TEXT,
    NO_PUNISHMENT_TEXT
)

def generate_strategy(
    agent_name: str,
    generation_number: int,
    inherited_strategies: str,
    punishment_mechanism: str,
    punishment_loss: float,
    initial_endowment: int,
    system_prompt: str,
    llm_type: str,
    client: Any
) -> tuple[str, str]:
    """
    Generate a strategy for an agent using LLM.
    
    Args:
        agent_name: Name of the agent
        generation_number: Current generation number
        inherited_strategies: Strategies from previous generation
        punishment_mechanism: Type of punishment mechanism
        punishment_loss: Punishment loss multiplier
        initial_endowment: Initial resource endowment
        system_prompt: System prompt for LLM
        llm_type: Type of LLM to use
        client: LLM client instance
        
    Returns:
        Tuple of (strategy_justification, strategy)
    """
    # Get appropriate punishment text
    if punishment_mechanism == "costly_punishment":
        punishment_text = COSTLY_PUNISHMENT_TEXT.format(punishment_loss=punishment_loss)
    elif punishment_mechanism == "partner_choice":
        punishment_text = PARTNER_CHOICE_PUNISHMENT_TEXT
    else:
        punishment_text = NO_PUNISHMENT_TEXT

    # Select and format appropriate prompt
    if generation_number == 1:
        prompt = INITIAL_STRATEGY_PROMPT.format(
            agent_name=agent_name,
            punishment_text=punishment_text
        )
    else:
        prompt = EVOLVED_STRATEGY_PROMPT.format(
            agent_name=agent_name,
            inherited_strategies=inherited_strategies,
            punishment_text=punishment_text
        )

    strategy_output = prompt_llm(prompt, llm_type=llm_type, system_prompt=system_prompt, client=client)
    parts = strategy_output.split("My strategy will be", 1)
    
    if len(parts) == 2:
        strategy_justification = parts[0].strip()
        strategy = "My strategy will be" + parts[1].strip()
    else:
        strategy_justification = ""
        strategy = strategy_output.strip()

    print(f"{agent_name}: \n Justification: {strategy_justification} \n Strategy: {strategy} ")
    return strategy_justification, strategy

def initialize_agents(
    num_agents: int,
    initial_endowment: int,
    generation_number: int,
    inherited_strategies: List[str],
    punishment_mechanism: str,
    punishment_loss: float,
    system_prompt: str,
    llm_type: str,
    client: Any
) -> List[Agent]:
    """
    Initialize a set of agents for the game.
    
    Args:
        num_agents: Number of agents to create
        initial_endowment: Initial resource endowment
        generation_number: Current generation number
        inherited_strategies: Strategies from previous generation
        punishment_mechanism: Type of punishment mechanism
        punishment_loss: Punishment loss multiplier
        system_prompt: System prompt for LLM
        llm_type: Type of LLM to use
        client: LLM client instance
        
    Returns:
        List of initialized agents
    """
    agents = []

    with ThreadPoolExecutor() as executor:
        futures = []
        for i in range(num_agents):
            name = f"{generation_number}_{i+1}"
            futures.append(
                executor.submit(
                    generate_strategy,
                    name,
                    generation_number,
                    inherited_strategies,
                    punishment_mechanism,
                    punishment_loss,
                    initial_endowment,
                    system_prompt,
                    llm_type,
                    client
                )
            )

        # Collect results and create agents
        for i, future in enumerate(futures):
            strategy_justification, new_strategy = future.result()
            name = f"{generation_number}_{i+1}"
            agents.append(
                Agent(
                    name=name,
                    reputation=False,
                    resources=initial_endowment,
                    strategy=new_strategy,
                    strategy_justification=strategy_justification
                )
            )

    random.shuffle(agents)
    return agents 