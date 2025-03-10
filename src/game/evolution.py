import os
from typing import List, Tuple, Any
import random
from src.models.agent import Agent
from src.models.simulation_data import SimulationData
from src.game.initialization import initialize_agents
from src.game.donor_game import donor_game
from src.game.pairing import bipartite_round_robin, extend_rounds
from src.game.selection import select_top_agents, select_random_agents, select_highest_reputation
from src.utils.helpers import save_simulation_data
from src.utils.logger import setup_logger
from src.utils.checkpoint import ExperimentState, save_checkpoint

# Set up logger for this module
logger = setup_logger(__name__)

def run_generations(
    experiment_state: ExperimentState,
    client: Any,
    checkpoint_dir: str = "checkpoints"
) -> Tuple[List[Agent], SimulationData]:
    """
    Run the evolutionary donor game for multiple generations.
    
    Args:
        experiment_state: Current state of the experiment
        client: LLM client instance
        checkpoint_dir: Directory to store checkpoints
        
    Returns:
        Tuple of (all agents, simulation data)
    """
    params = experiment_state.parameters
    all_agents = []
    all_donations = []
    all_average_final_resources = []
    conditional_survival = 0
    prev_gen_strategies = []

    # Initialize agents if starting fresh
    if not experiment_state.agents:
        agents = initialize_agents(
            params["num_agents"],
            params["initial_endowment"],
            1,
            ["No previous strategies"],
            params["punishment_mechanism"],
            params["punishment_loss"],
            params["system_prompt"],
            params["llm_type"],
            client
        )
        experiment_state.agents = agents
        all_agents.extend(agents)
    else:
        agents = experiment_state.agents
        all_agents.extend(agents)

    for i in range(experiment_state.current_generation, params["num_generations"]):
        generation_info = f"Generation {i + 1}"
        logger.info(generation_info)
        for agent in agents:
            agent.history.append(generation_info + ": \n")
            prev_gen_strategies.append(agent.strategy)
            if int(agent.name.split('_')[0]) == i-1:
                conditional_survival += 1

        # Create rounds using bipartiteRoundRobin
        initial_rounds = bipartite_round_robin(agents)
        rounds = extend_rounds(initial_rounds)

        try:
            generation_history, donation_records = donor_game(
                agents=agents,
                rounds=rounds,  # Pass all rounds at once
                generation=i+1,
                simulation_data=experiment_state.simulation_data,
                initial_endowment=params["initial_endowment"],
                cooperation_gain=params["cooperation_gain"],
                punishment_loss=params["punishment_loss"],
                punishment_mechanism=params["punishment_mechanism"]
            )
            all_donations.extend(donation_records)

            # Update and save checkpoint after each round
            experiment_state.current_generation = i
            experiment_state.current_game = 1
            experiment_state.current_round = 0
            experiment_state.agents = agents
            save_checkpoint(experiment_state, checkpoint_dir)

        except Exception as e:
            logger.error(f"Error in generation {i + 1}: {str(e)}")
            save_checkpoint(experiment_state, checkpoint_dir)
            raise

    # Save final simulation data
    save_simulation_data(
        experiment_state.simulation_data,
        params["save_path"],
        params["llm_type"],
        params["cooperation_gain"],
        params["punishment_loss"],
        params["reputation_mechanism"]
    )

    return all_agents, experiment_state.simulation_data 