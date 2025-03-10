# Donor Game with Cultural Evolution

This repo takes over code by https://github.com/aronvallinder/llm-donor-game/ and refactors the main ipynb into a more manageable codebase. 

The project implements an evolutionary donor game where agents powered by Large Language Models (LLMs) interact in a resource-sharing environment. The game explores how different strategies evolve over generations and how various mechanisms (reputation, punishment) affect cooperation.

## Game Description

In each round:
- Agents are paired randomly
- One agent is the donor, the other is the recipient
- The donor decides how much of their resources to give
- The recipient receives a multiplied amount of what was donated
- Agents can see limited history of their partners' previous interactions
- After each generation, the most successful agents survive and new ones are created

## Features

- Multiple LLM support (GPT-3.5/4, Claude, Gemini)
- Different selection mechanisms (top performers, random, reputation-based)
- Optional punishment mechanisms (costly punishment, partner choice)
- Reputation tracking
- Detailed logging and data collection

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd llm-donor-game
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a .env file in the root directory and add your API keys:
```bash
# .env
OPENAI_API_KEY="your-openai-key"
ANTHROPIC_API_KEY="your-anthropic-key"
GEMINI_API_KEY="your-gemini-key"
```

## Configuration

Edit `src/main.py` to configure the experiment parameters:

```python
# Game parameters
COOPERATION_GAIN = 2.0
PUNISHMENT_LOSS = 2.0
NUM_GENERATIONS = 2
NUM_AGENTS = 12
INITIAL_ENDOWMENT = 10

# Game mechanics
REPUTATION_MECHANISM = "three_last_traces"
PUNISHMENT_MECHANISM = "none"  # Options: "none", "costly_punishment", "partner_choice"
SELECTION_METHOD = "top"  # Options: "top", "random", "reputation"

# LLM configuration
LLM_TYPE = "gemini-2.0-flash"  # Options: "gpt-3.5-turbo", "gpt-4", "claude-3-opus", "claude-3-sonnet", "gemini-2.0-flash"
```

## Running the Experiment

### Command Line
```bash
python -m src.main
```

### VS Code
The project includes VS Code launch configurations. To run:

1. Open the project in VS Code
2. Go to the Run and Debug view (Ctrl+Shift+D / Cmd+Shift+D)
3. Select either:
   - "Donor Game" for normal execution
   - "Donor Game (Debug)" for debugging with access to library code
4. Press F5 or click the green play button

Results will be saved in the `results` directory as JSON files.

## Project Structure

```
llm-donor-game/
├── src/
│   ├── game/
│   │   ├── donor_game.py     # Core game logic
│   │   ├── evolution.py      # Generational evolution
│   │   ├── initialization.py # Agent initialization
│   │   ├── pairing.py       # Agent pairing logic
│   │   └── selection.py     # Selection strategies
│   ├── models/
│   │   ├── agent.py         # Agent data model
│   │   └── simulation_data.py # Simulation data model
│   ├── utils/
│   │   ├── helpers.py       # Utility functions
│   │   └── llm.py          # LLM interaction
│   └── main.py             # Main experiment runner
├── .vscode/                # VS Code configuration
│   └── launch.json        # Debug/run configurations
├── .env                    # API keys configuration
├── requirements.txt
└── README.md
```
