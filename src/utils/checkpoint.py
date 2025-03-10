import os
import json
import uuid
import hashlib
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

from src.models.agent import Agent
from src.models.simulation_data import SimulationData
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class ExperimentState:
    """Class to represent the state of an experiment for checkpointing."""
    def __init__(
        self,
        uuid: str,
        param_hash: str,
        current_generation: int,
        current_game: int,
        current_round: int,
        agents: list[Agent],
        simulation_data: SimulationData,
        parameters: Dict[str, Any]
    ):
        self.uuid = uuid
        self.param_hash = param_hash
        self.current_generation = current_generation
        self.current_game = current_game
        self.current_round = current_round
        self.agents = agents
        self.simulation_data = simulation_data
        self.parameters = parameters
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "uuid": self.uuid,
            "param_hash": self.param_hash,
            "current_generation": self.current_generation,
            "current_game": self.current_game,
            "current_round": self.current_round,
            "agents": [agent.__dict__ for agent in self.agents],
            "simulation_data": self.simulation_data.to_dict(),
            "parameters": self.parameters,
            "timestamp": self.timestamp
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentState':
        """Create ExperimentState instance from dictionary."""
        # Reconstruct Agent objects
        agents = []
        for agent_data in data["agents"]:
            agent = Agent(
                name=agent_data["name"],
                reputation=agent_data["reputation"],
                resources=agent_data["resources"],
                strategy=agent_data["strategy"],
                strategy_justification=agent_data["strategy_justification"]
            )
            agent.__dict__.update(agent_data)
            agents.append(agent)

        # Reconstruct SimulationData
        simulation_data = SimulationData(hyperparameters=data["simulation_data"]["hyperparameters"])
        simulation_data.__dict__.update(data["simulation_data"])

        return cls(
            uuid=data["uuid"],
            param_hash=data["param_hash"],
            current_generation=data["current_generation"],
            current_game=data["current_game"],
            current_round=data["current_round"],
            agents=agents,
            simulation_data=simulation_data,
            parameters=data["parameters"]
        )

def generate_param_hash(parameters: Dict[str, Any]) -> str:
    """Generate a hash from experiment parameters."""
    # Sort parameters to ensure consistent hash
    param_str = json.dumps(parameters, sort_keys=True)
    return hashlib.sha256(param_str.encode()).hexdigest()[:12]

def save_checkpoint(
    state: ExperimentState,
    checkpoint_dir: str = "checkpoints"
) -> str:
    """Save experiment state to checkpoint file."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(
        checkpoint_dir,
        f"checkpoint_{state.param_hash}_{state.uuid}.json"
    )
    
    with open(checkpoint_path, 'w') as f:
        json.dump(state.to_dict(), f, indent=2)
    
    logger.info(f"Saved checkpoint to {checkpoint_path}")
    return checkpoint_path

def find_latest_checkpoint(
    param_hash: str,
    checkpoint_dir: str = "checkpoints"
) -> Optional[str]:
    """Find the most recent checkpoint file for given parameters."""
    if not os.path.exists(checkpoint_dir):
        return None
        
    checkpoints = []
    for file in os.listdir(checkpoint_dir):
        if file.startswith(f"checkpoint_{param_hash}_") and file.endswith(".json"):
            path = os.path.join(checkpoint_dir, file)
            checkpoints.append((path, os.path.getmtime(path)))
    
    if not checkpoints:
        return None
        
    # Return the most recent checkpoint
    return max(checkpoints, key=lambda x: x[1])[0]

def load_checkpoint(checkpoint_path: str) -> ExperimentState:
    """Load experiment state from checkpoint file."""
    with open(checkpoint_path, 'r') as f:
        data = json.load(f)
    
    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    return ExperimentState.from_dict(data)

def get_or_create_experiment_state(
    parameters: Dict[str, Any],
    checkpoint_dir: str = "checkpoints"
) -> Tuple[ExperimentState, bool]:
    """
    Get existing experiment state or create new one.
    Returns tuple of (state, is_new).
    """
    param_hash = generate_param_hash(parameters)
    checkpoint_path = find_latest_checkpoint(param_hash, checkpoint_dir)
    
    if checkpoint_path:
        logger.info(f"Found existing checkpoint for parameters (hash: {param_hash})")
        return load_checkpoint(checkpoint_path), False
    
    # Create new experiment state
    experiment_uuid = str(uuid.uuid4())
    state = ExperimentState(
        uuid=experiment_uuid,
        param_hash=param_hash,
        current_generation=0,
        current_game=1,
        current_round=0,
        agents=[],  # Will be populated by evolution.py
        simulation_data=SimulationData(hyperparameters=parameters),
        parameters=parameters
    )
    
    logger.info(f"Created new experiment state (UUID: {experiment_uuid}, hash: {param_hash})")
    return state, True 