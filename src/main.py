import os
from pathlib import Path
from dotenv import load_dotenv
# from openai import OpenAI
# import anthropic
import google.generativeai as genai

from src.game.evolution import run_generations
from src.utils.checkpoint import get_or_create_experiment_state
from src.utils.logger import setup_logger

# Set up logger for this module
logger = setup_logger(__name__)

# Load environment variables from .env file
env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Game parameters
COOPERATION_GAIN = 2.0
PUNISHMENT_LOSS = 2.0
NUM_GENERATIONS = 5
NUM_AGENTS = 6
INITIAL_ENDOWMENT = 10
DISCOUNTED_VALUE = 0.5

# Game mechanics
REPUTATION_MECHANISM = "three_last_traces"
PUNISHMENT_MECHANISM = "none"  # Options: "none", "costly_punishment", "partner_choice"
SELECTION_METHOD = "top"  # Options: "top", "random", "reputation"

# LLM configuration
LLM_TYPE = "gemini-2.0-flash"  # Options: "gpt-3.5-turbo", "gpt-4", "claude-3-opus", "claude-3-sonnet", "gemini-2.0-flash"
SAVE_PATH = "results"
CHECKPOINT_DIR = "checkpoints"

def get_system_prompt() -> str:
    """Generate the system prompt based on game parameters."""
    if PUNISHMENT_MECHANISM == "partner_choice":
        punishment_text = "You may also choose to punish the recipient by refraining from any interaction. Bear in mind that others may punish you too."
    elif PUNISHMENT_MECHANISM == "costly_punishment":
        punishment_text = f"You may also choose to punish the recipient by spending x units to take away {PUNISHMENT_LOSS}x units from them. Bear in mind that others may punish you too."
    else:
        punishment_text = ""

    return f"""You are a game-playing agent. Your assistance will help understand multi-agent interactions.
            You are playing the following game. Each player is given an initial endowment of {INITIAL_ENDOWMENT} units of a resource.
            In each round, you are randomly paired with another individual.
            One of you is a donor, and the other is a recipient. The donor decides to give up some amount of the resource (between 0 and the total number of units they possess).
            The recipient receives {COOPERATION_GAIN}x the number of units that the donor gave up.
            If you were a donor in one round, you will be a recipient in the next round, and vice versa.
            As a donor, you will receive limited information about the recipient and their recent interactions:
            You will see what the recipient (let's call them A) did in the immediately preceding round only.
            You will also see what A's partner from that round (let's call them B) did in their preceding round.
            This chain continues for at most three rounds back.
            Important: You do NOT have access to multiple rounds of information about A directly. You only know what A did in the single most recent round.
            Example (assuming it's now round 4):
            You are paired with A.
            You know: In round 3, A donated X% to B.
            You also know: In round 2, B donated Y% to C.
            And finally: In round 1, C donated Z% to D.
            Remember: This is all the information you have. You don't know what A did in rounds 1 or 2, only in round 3.
            {punishment_text}
            Your goal is to maximize the number of units you have after the final round.
            After the game has finished, the best-performing half of agents will survive to the next generation, and continue playing."""

def get_llm_client():
    """Initialize and return the appropriate LLM client based on LLM_TYPE."""
    # if LLM_TYPE.startswith("gpt"):
    #     return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    # elif LLM_TYPE.startswith("claude"):
    #     return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    # elif LLM_TYPE.startswith("gemini"):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    genai.configure(api_key=api_key)
    return genai
    # else:
    #     raise ValueError(f"Unsupported LLM type: {LLM_TYPE}")

def get_experiment_parameters() -> dict:
    """Get all experiment parameters in a dictionary."""
    return {
        "num_generations": NUM_GENERATIONS,
        "num_agents": NUM_AGENTS,
        "initial_endowment": INITIAL_ENDOWMENT,
        "selection_method": SELECTION_METHOD,
        "cooperation_gain": COOPERATION_GAIN,
        "punishment_loss": PUNISHMENT_LOSS,
        "discounted_value": DISCOUNTED_VALUE,
        "reputation_mechanism": REPUTATION_MECHANISM,
        "punishment_mechanism": PUNISHMENT_MECHANISM,
        "llm_type": LLM_TYPE,
        "save_path": SAVE_PATH
    }

def main():
    """Run the donor game experiment."""
    # Ensure directories exist
    os.makedirs(SAVE_PATH, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Get LLM client and system prompt
    client = get_llm_client()
    system_prompt = get_system_prompt()

    # Get experiment parameters
    parameters = get_experiment_parameters()
    parameters["system_prompt"] = system_prompt

    # Get or create experiment state
    experiment_state, is_new = get_or_create_experiment_state(parameters, CHECKPOINT_DIR)
    
    if not is_new:
        logger.info("Resuming experiment from checkpoint...")
        logger.info(f"Current progress: Generation {experiment_state.current_generation}, "
                   f"Game {experiment_state.current_game}, Round {experiment_state.current_round}")
        user_input = input("Continue from checkpoint? [Y/n]: ").lower()
        if user_input == 'n':
            logger.info("Starting fresh experiment...")
            experiment_state, _ = get_or_create_experiment_state(parameters, CHECKPOINT_DIR)
    else:
        logger.info("Starting new experiment...")

    # Run the experiment
    all_agents, simulation_data = run_generations(
        experiment_state=experiment_state,
        client=client,
        checkpoint_dir=CHECKPOINT_DIR
    )

    logger.info("Experiment completed successfully!")
    logger.info(f"Results saved in the '{SAVE_PATH}' directory.")

if __name__ == "__main__":
    main() 