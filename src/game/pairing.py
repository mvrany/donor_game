from typing import List, Tuple, Dict, Any
from threading import Lock
from queue import Queue
import textwrap

from src.models.agent import Agent
from src.models.simulation_data import AgentRoundData
from src.utils.llm import prompt_llm
from src.utils.helpers import get_last_three_reversed
from src.utils.logger import setup_logger

# Set up logger for this module
logger = setup_logger(__name__)

def bipartite_round_robin(agents: List[Agent]) -> List[List[Tuple[Agent, Agent]]]:
    """
    Generate round-robin tournament pairings for agents.
    
    Args:
        agents: List of agents to pair
        
    Returns:
        List of rounds, where each round is a list of (donor, recipient) pairs
    """
    num_agents = len(agents)
    assert num_agents % 2 == 0, "Number of agents must be even."
    
    group_A = agents[:num_agents // 2]
    group_B = agents[num_agents // 2:]
    rounds = []
    toggle_roles = False
    
    for i in range(len(group_A)):
        # Rotate group B
        rotated_group_B = group_B[-i:] + group_B[:-i]
        if toggle_roles:
            round_pairings = list(zip(rotated_group_B, group_A))
        else:
            round_pairings = list(zip(group_A, rotated_group_B))
        rounds.append(round_pairings)
        toggle_roles = not toggle_roles
        
    return rounds

def extend_rounds(original_rounds: List[List[Tuple[Agent, Agent]]]) -> List[List[Tuple[Agent, Agent]]]:
    """
    Extend the rounds by adding reversed pairings.
    
    Args:
        original_rounds: Original list of round pairings
        
    Returns:
        Extended list of rounds including reversed pairings
    """
    extended_rounds = original_rounds.copy()
    for round in original_rounds:
        reversed_round = [(b, a) for a, b in round]
        extended_rounds.append(reversed_round)
    return extended_rounds

def calculate_received_amount(
    punishment_mechanism: str,
    refused: bool,
    cooperation_gain: float,
    response: float,
    punishment_loss: float,
    action: str = None
) -> float:
    """
    Calculate the amount received by the recipient based on the donor's action.
    
    Args:
        punishment_mechanism: Type of punishment mechanism
        refused: Whether the donor refused to play
        cooperation_gain: Multiplier for donations
        response: Amount given/used by donor
        punishment_loss: Multiplier for punishment
        action: Type of action taken ('donate' or 'punish')
        
    Returns:
        Amount received by the recipient
    """
    if punishment_mechanism == "partner_choice":
        return cooperation_gain * response if not refused else 0
    elif punishment_mechanism == "costly_punishment":
        if action is None:
            raise ValueError("Action must be specified for costly_punishment mechanism")
        if action == 'donate':
            return cooperation_gain * response
        elif action == 'punish':
            return -punishment_loss * response
        else:
            raise ValueError(f"Unknown action for costly_punishment: {action}")
    elif punishment_mechanism == 'none':
        return cooperation_gain * response
    else:
        raise ValueError(f"Unknown punishment mechanism: {punishment_mechanism}")

def handle_refuse_action(
    donor: Agent,
    recipient: Agent,
    round_index: int,
    recipient_behavior: str,
    justification: str
) -> str:
    """Handle the case where donor refuses to play."""
    action_info = (
        f"{donor.name} refused to play with {recipient.name}.\n"
        f"Resources: {donor.name}: {donor.resources:.2f} and {recipient.name}: {recipient.resources:.2f} \n"
        f"Recipient traces: {recipient_behavior} \n"
        f"Justification:\n{textwrap.fill(justification, width=80, initial_indent='    ', subsequent_indent='    ')}\n"
    )
    new_trace = recipient.traces[-1].copy() if recipient.traces else []
    new_trace.append(f"In round {round_index + 1}, {donor.name} refused to play with {recipient.name}.")
    donor.traces.append(new_trace)
    logger.debug(action_info)
    return action_info

def handle_donate_action(
    donor: Agent,
    recipient: Agent,
    response: float,
    round_index: int,
    recipient_behavior: str,
    justification: str,
    cooperation_gain: float,
    discounted_value: float = 0.5
) -> str:
    """Handle the case where donor donates resources."""
    percentage_donated = response / donor.resources if donor.resources != 0 else 1
    donor.resources -= response
    donor.total_donated += response
    donor.potential_donated += donor.resources + response
    recipient.resources += cooperation_gain * response
    
    action_info = (
        f"{donor.name}: -{response} ({percentage_donated:.2%}) and {recipient.name}: +{cooperation_gain * response}.\n"
        f"Previous resources: {donor.name}: {donor.resources+response:.2f} and {recipient.name}: {recipient.resources-(cooperation_gain* response)}.\n"
        f"New resources: {donor.name}: {donor.resources:.2f} and {recipient.name}: {recipient.resources:.2f}.\n"
        f"Recipient traces: {recipient_behavior}"
        f"Justification:\n{textwrap.fill(justification, width=80, initial_indent='    ', subsequent_indent='    ')}\n"
    )

    new_trace = recipient.traces[-1].copy() if recipient.traces else []
    new_trace.append(f"In round {round_index + 1}, {donor.name} donated {percentage_donated * 100:.2f}% of their resources to {recipient.name}.")
    donor.traces.append(new_trace)

    if donor.reputation == False:
        donor.reputation = percentage_donated
    else:
        donor.reputation = ((1 - abs(percentage_donated - recipient.reputation)) + discounted_value * donor.reputation) / (1 + discounted_value)

    logger.debug(action_info)
    return action_info

def handle_punish_action(
    donor: Agent,
    recipient: Agent,
    response: float,
    round_index: int,
    recipient_behavior: str,
    justification: str,
    punishment_loss: float
) -> str:
    """Handle the case where donor punishes recipient."""
    percentage_donated = response / donor.resources if donor.resources != 0 else 1
    donor.resources -= response
    donor.total_donated += response
    donor.potential_donated += donor.resources + response
    recipient.resources = max(0, recipient.resources - punishment_loss * response)
    
    action_info = (
        f"{donor.name}: -{response} ({percentage_donated:.2%}) and {recipient.name}: - {punishment_loss * response}.\n"
        f"Previous resources: {donor.name}: {donor.resources+response:.2f} and {recipient.name}: {recipient.resources+(punishment_loss* response)}."
        f"New resources: {donor.name}: {donor.resources:.2f} and {recipient.name}: {recipient.resources:.2f}.\n"
        f"Recipient traces: {recipient_behavior} \n"
        f"Justification:\n{textwrap.fill(justification, width=80, initial_indent='    ', subsequent_indent='    ')}\n"
    )

    new_trace = recipient.traces[-1].copy() if recipient.traces else []
    new_trace.append(f"In round {round_index + 1}, {donor.name} punished {recipient.name} by spending {response} units to take away {punishment_loss * response} units from their resources.")
    donor.traces.append(new_trace)

    logger.debug(action_info)
    return action_info

def handle_invalid_action(
    donor: Agent,
    recipient: Agent,
    round_index: int,
    recipient_behavior: str,
    justification: str
) -> str:
    """Handle the case where donor attempts an invalid action."""
    action_info = (
        f"{donor.name} attempted an invalid action.\n"
        f"Resources: {donor.name}: {donor.resources:.2f} and {recipient.name}: {recipient.resources:.2f} \n"
        f"Recipient traces: {recipient_behavior} \n"
        f"Justification:\n{textwrap.fill(justification, width=80, initial_indent='    ', subsequent_indent='    ')}\n"
    )
    logger.debug(action_info)
    return action_info

def create_agent_round_data(
    agent: Agent,
    round_index: int,
    game_number: int,
    paired_with: str,
    generation: int,
    donated: float,
    received: float,
    is_donor: bool,
    punished: bool,
    justification: str
) -> AgentRoundData:
    """Create a data record for an agent's round."""
    return AgentRoundData(
        agent_name=agent.name,
        round_number=round_index + 1,
        game_number=game_number,
        paired_with=paired_with,
        current_generation=generation,
        resources=agent.resources,
        donated=donated,
        received=received,
        strategy=agent.strategy,
        strategy_justification=agent.strategy_justification,
        reputation=agent.reputation,
        is_donor=is_donor,
        traces=agent.traces,
        history=agent.history,
        justification=justification,
        punished=punished
    )

def handle_pairing_thread_safe(
    donor: Agent,
    recipient: Agent,
    round_index: int,
    generation: int,
    game_number: int,
    agent_locks: Dict[str, Lock],
    donation_records: Queue,
    agent_updates: Queue,
    cooperation_gain: float,
    punishment_loss: float,
    punishment_mechanism: str
) -> Tuple[str, AgentRoundData, AgentRoundData]:
    """
    Handle a single donor-recipient pairing in a thread-safe manner.
    
    Args:
        donor: The donating agent
        recipient: The receiving agent
        round_index: Current round number
        generation: Current generation number
        game_number: Current game number
        agent_locks: Dictionary of locks for each agent
        donation_records: Queue for storing donation records
        agent_updates: Queue for storing agent updates
        cooperation_gain: Multiplier for donations
        punishment_loss: Multiplier for punishment
        punishment_mechanism: Type of punishment mechanism
        
    Returns:
        Tuple containing action info and data for both agents
    """
    logger.debug(f"Starting pairing: Donor={donor.name}, Recipient={recipient.name}, Round={round_index + 1}, Game={game_number}")
    
    action_info = ""
    punished = False
    action = 'donate'
    justification = ""
    response = 0

    recipient_behavior = ""
    if donor.traces:
        last_trace = recipient.traces[-1]
        if isinstance(last_trace, list):
            recipient_behavior = get_last_three_reversed(last_trace)
        else:
            recipient_behavior = str(last_trace)

    with agent_locks[donor.name], agent_locks[recipient.name]:
        prompt = generate_donor_prompt(
            donor, generation, round_index + 1, recipient,
            punishment_mechanism, cooperation_gain, punishment_loss
        )
        logger.debug(f"Generated prompt for {donor.name}:\n{prompt}")

        valid_response = False
        max_attempts = 10
        attempts = 0

        while not valid_response and attempts < max_attempts:
            try:
                full_response = prompt_llm(prompt, timeout=30)
                logger.debug(f"LLM response for {donor.name}:\n{full_response}")
                
                parts = full_response.split('Answer:', 1)

                if len(parts) == 2:
                    justification = parts[0].replace('Justification:', '').strip()
                    answer_part = parts[1].strip()

                    if punishment_mechanism == "partner_choice":
                        if "refuse" in answer_part.lower():
                            action = 'refuse'
                            response = 0
                            valid_response = True
                        else:
                            try:
                                response = float(answer_part.strip().split()[0])
                                action = 'donate'
                                valid_response = True
                            except (ValueError, IndexError):
                                pass

                    elif punishment_mechanism == "costly_punishment":
                        import re
                        match = re.search(r'(donate|punish).*?(\\d+(?:[.,]\\d+)?)', answer_part, re.IGNORECASE)
                        if match:
                            action = match.group(1).lower()
                            response = float(match.group(2).replace(',', '.'))
                            valid_response = True

                    else:  # No punishment mechanism
                        try:
                            response = float(answer_part.strip().split()[0])
                            action = 'donate'
                            valid_response = True
                        except (ValueError, IndexError):
                            pass

                if not valid_response:
                    logger.warning(f"Invalid response from {donor.name} in round {round_index + 1}. Retrying...")
                    attempts += 1

            except Exception as e:
                logger.error(f"Error processing response from {donor.name} in round {round_index + 1}: {str(e)}")
                attempts += 1

        if not valid_response:
            logger.warning(f"Failed to get a valid response from {donor.name} after {max_attempts} attempts")
            action = 'donate'
            response = 0

    # Process the action and update agent states
    logger.debug(f"Processing action: {action} from {donor.name} with response={response}")
    
    if action == 'refuse':
        action_info = handle_refuse_action(donor, recipient, round_index, recipient_behavior, justification)
    elif 0 <= response <= donor.resources:
        if action == 'donate':
            action_info = handle_donate_action(donor, recipient, response, round_index, recipient_behavior, justification, cooperation_gain)
        elif action == 'punish':
            action_info = handle_punish_action(donor, recipient, response, round_index, recipient_behavior, justification, punishment_loss)
    else:
        action_info = handle_invalid_action(donor, recipient, round_index, recipient_behavior, justification)

    # Create round data for both agents
    donor_data = create_agent_round_data(donor, round_index, game_number, recipient.name, generation, response, 0, True, punished, justification)
    recipient_data = create_agent_round_data(
        recipient, round_index, game_number, donor.name, generation, 0,
        calculate_received_amount(punishment_mechanism, action == 'refuse', cooperation_gain, response, punishment_loss, action),
        False, False, ""
    )

    logger.debug(f"Completed pairing: Donor={donor.name}, Recipient={recipient.name}, Round={round_index + 1}, Game={game_number}")
    return action_info, donor_data, recipient_data

def generate_donor_prompt(
    donor: Agent,
    generation: int,
    round: int,
    recipient: Agent,
    punishment_mechanism: str,
    cooperation_gain: float,
    punishment_loss: float
) -> str:
    """
    Generate the prompt for the game interaction.
    
    Args:
        donor: The donating agent
        generation: Current generation number
        round: Current round number
        recipient: The receiving agent
        punishment_mechanism: Type of punishment mechanism
        cooperation_gain: Multiplier for donations
        punishment_loss: Multiplier for punishment
        
    Returns:
        Formatted prompt string for the LLM
    """
    strategy_text = f"As you will recall, here is the strategy you decided to follow: {donor.strategy}" if donor.strategy else ""
    
    if punishment_mechanism == "partner_choice":
        punishment_text = f"If you do not wish to play with {recipient.name}, simply state: Refuse to play. Otherwise, state your answer as a real number between 0 and {donor.resources} (inclusive)."
        punishment_text_2 = ""
    elif punishment_mechanism == "costly_punishment":
        punishment_text = f"State your answer as EITHER Donate, x OR Punish, x, to either have them receive {cooperation_gain}x units or punish them by taking away {punishment_loss}x of their units."
        punishment_text_2 = f"Remember, state your answer as EITHER Donate, x OR Punish, x."
    else:
        punishment_text = f"State your answer as a real number between 0 and {donor.resources} (inclusive)."
        punishment_text_2 = ""

    recipient_behavior = ""
    if donor.traces:
        last_trace = recipient.traces[-1]
        if isinstance(last_trace, list):
            recipient_behavior = get_last_three_reversed(last_trace)
        else:
            recipient_behavior = str(last_trace)

    return (
        f"Your name is {donor.name}. "
        f"{strategy_text}"
        f"This is generation {generation}. It is now round {round}. "
        f"In this round, you have been paired with {recipient.name}. They currently have {recipient.resources} units of the valuable resource."
        f"{recipient_behavior}"
        f"You will take the role of donor. You currently have {donor.resources} units of the valuable resource. "
        f"How many units do you give up?"
        f"{punishment_text}"
        "Very briefly think step by step about how you apply your strategy in this situation and then provide your answer."
        "State your full response in the following form:"
        "Justification:"
        "[multi-line justification here]"
        " "
        "Answer: [answer here]"
        "Remember: First state your justification and after that your answer."
        f"{punishment_text_2}"
    ) 