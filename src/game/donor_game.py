from typing import List, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from queue import Queue
import time
import random

from src.models.agent import Agent
from src.models.simulation_data import SimulationData, AgentRoundData
from src.game.pairing import handle_pairing_thread_safe
from src.utils.token_counter import token_counter
from src.utils.logger import setup_logger

# Set up logger for this module
logger = setup_logger(__name__)

# Rate limiting settings
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 1.0  # seconds
MAX_WORKERS = 4  # Reduce concurrent API calls

def donor_game(
    agents: List[Agent],
    rounds: List[List[Tuple[Agent, Agent]]],
    generation: int,
    simulation_data: SimulationData,
    initial_endowment: int,
    cooperation_gain: float,
    punishment_loss: float = 0,
    punishment_mechanism: str = "none"
) -> Tuple[List[str], List[Any]]:
    """
    Run the donor game for a single generation.
    
    Args:
        agents: List of agents participating in the game
        rounds: List of round pairings
        generation: Current generation number
        simulation_data: Object to store simulation data
        initial_endowment: Starting resources for each agent
        cooperation_gain: Multiplier for donations
        punishment_loss: Multiplier for punishment (if enabled)
        punishment_mechanism: Type of punishment mechanism ("none", "costly_punishment", or "partner_choice")
    
    Returns:
        Tuple containing full history and donation records
    """
    full_history = []
    donation_records = Queue()
    agent_updates = Queue()
    total_rounds = len(rounds) * 2  # Total rounds across both games

    # Reset token counter for this generation
    token_counter.reset()

    # Create locks for each agent
    agent_locks = {agent.name: Lock() for agent in agents}

    def handle_pairing_with_retry(*args, **kwargs) -> Tuple[str, AgentRoundData, AgentRoundData]:
        """Wrapper to handle API calls with retries and exponential backoff"""
        retry_count = 0
        while retry_count < MAX_RETRIES:
            try:
                return handle_pairing_thread_safe(*args, **kwargs)
            except Exception as e:
                retry_count += 1
                if "429" in str(e):  # Rate limit error
                    delay = INITIAL_RETRY_DELAY * (2 ** retry_count) + random.uniform(0, 1)
                    logger.warning(f"Rate limit hit, retrying in {delay:.2f} seconds (attempt {retry_count}/{MAX_RETRIES})")
                    time.sleep(delay)
                else:
                    logger.error(f"Unexpected error in pairing: {str(e)}")
                    if retry_count == MAX_RETRIES:
                        raise
        
        # If we get here, we've exhausted retries
        logger.error("Max retries exceeded, returning default values")
        donor, recipient = args[0], args[1]
        return "", AgentRoundData(), AgentRoundData()

    def play_game(game_number: int, game_rounds: List[List[Tuple[Agent, Agent]]]) -> Dict[int, List[str]]:
        round_results = {i: [] for i in range(len(game_rounds))}

        for round_index, round_pairings in enumerate(game_rounds):
            if round_index == 0:
                # Initialize traces for the first round
                for agent in agents:
                    agent.traces = [[f"{agent.name} did not have any previous interactions."]]

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = []
                for donor, recipient in round_pairings:
                    if round_index > 0:
                        donor.traces.append(recipient.traces[-1].copy())
                    future = executor.submit(
                        handle_pairing_with_retry,
                        donor, recipient, round_index, generation, game_number,
                        agent_locks, donation_records, agent_updates,
                        cooperation_gain, punishment_loss, punishment_mechanism
                    )
                    futures.append(future)

                for future in as_completed(futures):
                    try:
                        action_info, donor_data, recipient_data = future.result()
                        if action_info:
                            round_results[round_index].append(action_info)
                        if donor_data and recipient_data:
                            simulation_data.agents_data.append(donor_data.__dict__)
                            simulation_data.agents_data.append(recipient_data.__dict__)
                    except Exception as e:
                        logger.error(f"Error processing future result: {str(e)}")
            
            # Log round completion
            current_round = round_index + 1 + (len(game_rounds) * (game_number - 1))
            logger.info(f"Round {current_round}/{total_rounds} completed (Generation: {generation}, Game: {game_number})")

        return round_results

    # Play the first game
    logger.info(f"Starting Game 1 of Generation {generation}")
    game1_results = play_game(1, rounds)

    # Report token usage after Game 1
    game1_tokens = token_counter.get_total_counts()
    logger.info(f"Game 1 Token Usage: Input tokens: {game1_tokens.input_tokens:,}, Output tokens: {game1_tokens.output_tokens:,}")

    # Compile results for Game 1
    for round_index in range(len(rounds)):
        full_history.append(f"Round {round_index + 1} (Game 1):\n")
        full_history.extend(game1_results[round_index])

    # Apply updates after all threads have completed
    while not agent_updates.empty():
        agent, history = agent_updates.get()
        agent.history.append(history)

    # Calculate and store average resources for Game 1
    average_resources_game1 = sum(agent.resources for agent in agents) / len(agents)
    logger.info(f"Average final resources for this generation (Game 1): {average_resources_game1:.2f}")

    # Store Game 1 final reputations
    game1_reputations = {agent.name: agent.reputation for agent in agents}

    # Reset resources, reputation, and history for Game 2
    for agent in agents:
        agent.resources = initial_endowment
        agent_generation = int(agent.name.split('_')[0])
        if agent_generation < generation:  # This is a surviving agent
            agent.reputation = agent.average_reputation
            agent.traces = agent.old_traces
        else:
            agent.reputation = False
            agent.traces.clear()
        agent.history.clear()

    # Reset token counter for Game 2
    token_counter.reset()

    # Generate pairings for Game 2 (reverse all pairs)
    reversed_rounds = [[tuple(reversed(pair)) for pair in round_pairings] for round_pairings in rounds]

    # Play the second game
    logger.info(f"Starting Game 2 of Generation {generation}")
    game2_results = play_game(2, reversed_rounds)

    # Report token usage after Game 2
    game2_tokens = token_counter.get_total_counts()
    logger.info(f"Game 2 Token Usage: Input tokens: {game2_tokens.input_tokens:,}, Output tokens: {game2_tokens.output_tokens:,}")

    # Store token usage in simulation data
    total_input_tokens = game1_tokens.input_tokens + game2_tokens.input_tokens
    total_output_tokens = game1_tokens.output_tokens + game2_tokens.output_tokens
    simulation_data.token_usage = {
        f"generation_{generation}": {
            "game1": game1_tokens.__dict__,
            "game2": game2_tokens.__dict__,
            "total": {
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "total_tokens": total_input_tokens + total_output_tokens
            }
        }
    }

    # Compile results for Game 2
    for round_index in range(len(reversed_rounds)):
        full_history.append(f"Round {round_index + 1} (Game 2):\n")
        full_history.extend(game2_results[round_index])

    # Apply updates after all threads have completed
    while not agent_updates.empty():
        agent, history = agent_updates.get()
        agent.history.append(history)

    # Calculate and store average resources for Game 2
    average_resources_game2 = sum(agent.resources for agent in agents) / len(agents)
    logger.info(f"Average final resources for this generation (Game 2): {average_resources_game2:.2f}")

    # Calculate final scores and reputations
    for agent in agents:
        agent.total_final_score = sum(agent.resources for _ in range(2))
        agent.average_reputation = (game1_reputations[agent.name] + agent.reputation) / 2 if agent.reputation is not False else game1_reputations[agent.name]

    return full_history, list(donation_records.queue) 