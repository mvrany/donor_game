from typing import Union, List, Any
import datetime
import os
import json
from src.models.simulation_data import SimulationData
from src.utils.logger import setup_logger

# Set up logger for this module
logger = setup_logger(__name__)

def get_last_three_reversed(item: Union[List[str], str, Any]) -> str:
    """
    Get the last three items from a list in reverse order, or convert a single item to string.
    
    Args:
        item: List of strings or single item
        
    Returns:
        String representation of the last three items or the single item
    """
    if isinstance(item, list):
        return " ".join(item[-3:][::-1])  # Returns last 3 items in reverse order
    elif isinstance(item, str):
        return item
    else:
        return str(item)

def save_simulation_data(
    simulation_data: SimulationData,
    folder_path: str,
    llm_type: str,
    cooperation_gain: float,
    punishment_loss: float,
    reputation_mechanism: str
) -> None:
    """
    Save simulation data to a JSON file.
    
    Args:
        simulation_data: The simulation data to save
        folder_path: Path to save the file
        llm_type: Type of LLM used
        cooperation_gain: Cooperation gain multiplier
        punishment_loss: Punishment loss multiplier
        reputation_mechanism: Type of reputation mechanism used
    """
    # Get the current timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Extract hyperparameters for the file name
    params = simulation_data.hyperparameters
    num_generations = params.get('numGenerations')
    num_agents = params.get('numAgents')
    selection_method = params.get('selectionMethod')

    # Create an informative file name
    filename = (
        f"Donor_Game_{llm_type}_coopGain_{cooperation_gain}"
        f"punLoss_{punishment_loss}_{reputation_mechanism}"
        f"gen{num_generations}_agents{num_agents}_{selection_method}_{timestamp}.json"
    )

    # Function to make data JSON serializable
    def make_serializable(obj: Any) -> Any:
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: make_serializable(value) for key, value in obj.items()}
        elif hasattr(obj, '__dict__'):
            return make_serializable(obj.__dict__)
        else:
            return str(obj)

    # Apply the serialization function to the entire data dictionary
    serializable_data = make_serializable(simulation_data.to_dict())

    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)

    # Create the full file path
    full_file_path = os.path.join(folder_path, filename)

    # Write the JSON data to the file
    with open(full_file_path, 'w') as f:
        json.dump(serializable_data, f, indent=4)

    logger.info(f"Simulation data saved to: {full_file_path}") 