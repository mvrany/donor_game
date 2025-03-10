# Initial strategy generation prompt for new agents (Generation 1)
# Purpose: Used to generate the initial strategy for agents in the first generation
# without any prior knowledge of successful strategies
INITIAL_STRATEGY_PROMPT = """
Your name is {agent_name}. 
Based on the description of the game, create a strategy that you will follow in the game. 
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
In the first round, there will be no information about the recipient's previous behavior - think carefully about how much to donate in that case.
Before formulating your strategy, briefly think step by step about what would be a successful strategy in this game.
Then describe your strategy briefly without explanation in one sentence that starts: My strategy will be.
"""

# Evolved strategy generation prompt for subsequent generations
# Purpose: Used to generate strategies for agents in later generations,
# taking into account successful strategies from previous generations
EVOLVED_STRATEGY_PROMPT = """
Your name is {agent_name}. 
How would you approach the game? Here is the advice of the best-performing 50% of the previous generation, along with their final scores.
Modify this advice to create your own strategy.
{inherited_strategies}
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
In the first round, there will be no information about the recipient's previous behavior - think carefully about how much to donate in that case.
Before formulating your strategy, briefly think step by step about what would be a successful strategy in this game. In particular, think about how you can improve on the surviving agents' strategies.
Then describe your strategy briefly without explanation in one sentence that starts: My strategy will be.
"""

# Game interaction prompt for each round
# Purpose: Used in each round to determine how much an agent will donate/punish
# based on their strategy and the recipient's history
GAME_INTERACTION_PROMPT = """
Your name is {donor_name}. 
{strategy_text}
This is generation {generation}. It is now round {round}. 
In this round, you have been paired with {recipient_name}. They currently have {recipient_resources} units of the valuable resource.
{recipient_behavior}
You will take the role of donor. You currently have {donor_resources} units of the valuable resource. 
How many units do you give up?
{punishment_text}
Very briefly think step by step about how you apply your strategy in this situation and then provide your answer.
State your full response in the following form:
Justification:
[multi-line justification here]
 
Answer: [answer here]
Remember: First state your justification and after that your answer.
{punishment_text_2}
"""

# Punishment text variations based on mechanism
# Purpose: Different punishment mechanisms require different instructions
COSTLY_PUNISHMENT_TEXT = """You may also choose to punish the recipient by spending x units to take away {punishment_loss}x of their resources. Bear in mind that others may punish you too."""

PARTNER_CHOICE_PUNISHMENT_TEXT = """You may also choose to punish the recipient by refraining from any interaction. Bear in mind that others may punish you too."""

NO_PUNISHMENT_TEXT = ""

# Game interaction punishment instructions
# Purpose: Specific instructions for how to format punishment responses
PARTNER_CHOICE_INSTRUCTION = """If you do not wish to play with {recipient_name}, simply state: Refuse to play. Otherwise, state your answer as a real number between 0 and {donor_resources} (inclusive)."""

COSTLY_PUNISHMENT_INSTRUCTION = """State your answer as EITHER Donate, x OR Punish, x, to either have them receive {cooperation_gain}x units or punish them by taking away {punishment_loss}x of their units."""

NO_PUNISHMENT_INSTRUCTION = """State your answer as a real number between 0 and {donor_resources} (inclusive).""" 