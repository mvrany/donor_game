from typing import List, Any
from concurrent.futures import ThreadPoolExecutor
import random

from ..models.agent import Agent
from ..utils.llm import prompt_llm

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
    if punishment_mechanism == "costly_punishment":
        punishment_text = f"You may also choose to punish the recipient by spending x units to take away {punishment_loss}x of their resources. Bear in mind that others may punish you too."
    elif punishment_mechanism == "partner_choice":
        punishment_text = "You may also choose to punish the recipient by refraining from any interaction. Bear in mind that others may punish you too."
    else:
        punishment_text = ""

    if generation_number == 1:
        prompt = (
            f"Your name is {agent_name}. "
            "Based on the description of the game, create a strategy that you will follow in the game. "
            "As a donor, you will receive limited information about the recipient and their recent interactions:"
            "You will see what the recipient (let's call them A) did in the immediately preceding round only."
            "You will also see what A's partner from that round (let's call them B) did in their preceding round."
            "This chain continues for at most three rounds back."
            "Important: You do NOT have access to multiple rounds of information about A directly. You only know what A did in the single most recent round."
            "Example (assuming it's now round 4):"
            "You are paired with A."
            "You know: In round 3, A donated X% to B."
            "You also know: In round 2, B donated Y% to C."
            "And finally: In round 1, C donated Z% to D."
            "Remember: This is all the information you have. You don't know what A did in rounds 1 or 2, only in round 3."
            f"{punishment_text}"
            "In the first round, there will be no information about the recipient's previous behavior - think carefully about how much to donate in that case."
            "Before formulating your strategy, briefly think step by step about what would be a successful strategy in this game."
            "Then describe your strategy briefly without explanation in one sentence that starts: My strategy will be."
        )
    else:
        prompt = (
            f"Your name is {agent_name}. "
            "How would you approach the game? Here is the advice of the best-performing 50% of the previous generation, along with their final scores."
            "Modify this advice to create your own strategy."
            f"{inherited_strategies}"
            "As a donor, you will receive limited information about the recipient and their recent interactions:"
            "You will see what the recipient (let's call them A) did in the immediately preceding round only."
            "You will also see what A's partner from that round (let's call them B) did in their preceding round."
            "This chain continues for at most three rounds back."
            "Important: You do NOT have access to multiple rounds of information about A directly. You only know what A did in the single most recent round."
            "Example (assuming it's now round 4):"
            "You are paired with A."
            "You know: In round 3, A donated X% to B."
            "You also know: In round 2, B donated Y% to C."
            "And finally: In round 1, C donated Z% to D."
            "Remember: This is all the information you have. You don't know what A did in rounds 1 or 2, only in round 3."
            f"{punishment_text}"
            "In the first round, there will be no information about the recipient's previous behavior - think carefully about how much to donate in that case."
            "Before formulating your strategy, briefly think step by step about what would be a successful strategy in this game. In particular, think about how you can improve on the surviving agents' strategies."
            "Then describe your strategy briefly without explanation in one sentence that starts: My strategy will be."
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