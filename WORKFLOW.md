# System Overview

## **The Big Picture**
Affine is a decentralized AI validation system built on the Bittensor network. It tests AI models (called "miners") by giving them challenges across different problem domains and uses a sophisticated scoring system called GRPO to rank them fairly while preventing cheating.

## **1. How Requests Are Sent**

**Discovery Phase:**
- The system first connects to the Bittensor blockchain to discover available miners
- Each miner has registered their AI model on the network with a unique identifier (UID)
- The system fetches information about each miner, including what AI model they're running

**Request Flow:**
- When testing a miner, the system sends HTTP requests to the Chutes.ai API (which acts as a gateway to the AI models)
- Each request contains the challenge prompt and specifies which model to use
- The system includes retry logic with exponential backoff - if a request fails, it waits a bit and tries again
- Multiple requests can be sent in parallel to test different miners simultaneously
- Each response includes the AI's answer, how long it took, and any errors that occurred

**Technical Details:**
- Uses aiohttp for async HTTP requests to handle many miners at once
- Has configurable timeouts (default 600 seconds) since some problems take time to solve
- Includes authentication tokens for API access
- Tracks latency and attempt counts for performance monitoring

## **2. How Examples Are Generated**

The system has three different problem environments, each generating challenges differently:

**SAT (Boolean Satisfiability):**
- Creates logic puzzles with variables that can be True or False
- First generates a random solution (like x1=True, x2=False, x3=True)
- Then creates logical constraints that this solution satisfies
- Builds a formula like "(x1 ∨ x3) ∧ (¬x2 ∨ x1)" and asks the AI to find values that make it true
- The challenge knows the correct answer in advance

**COIN (Coin Flip Guessing):**
- Simply picks HEADS or TAILS randomly
- Asks the AI to guess the result
- This tests basic instruction following and format compliance

**ABD (Abductive Reasoning - Code Input Generation):**
- Uses a dataset of Python programs with known inputs/outputs
- Takes a program and its expected output
- Asks the AI to figure out what input would produce that output
- Tests the AI's ability to reason backwards from effects to causes
- More complex because it involves code execution and reasoning

**Smart Sample Management:**
- The system pre-generates challenges and stores them in files (JSONL format)
- Maintains a "stock" of ready-to-use examples (default: 50 per environment)
- When stock runs low, automatically generates more in the background
- This prevents delays during evaluation and ensures fresh challenges
- Each challenge is used only once to prevent AIs from memorizing answers

## **3. How Evaluation Scores Are Calculated**

The system uses a sophisticated multi-layered scoring approach:

**Individual Challenge Scoring:**
- Each challenge gets a simple 0.0 or 1.0 score (wrong or right)
- For SAT: Checks if the proposed variable assignments actually satisfy all logical constraints
- For COIN: Checks if the guess matches the randomly chosen answer
- For ABD: Runs the generated input through the program and checks if output matches expected result

**GRPO (Group Relative Performance Optimization) Scoring:**
This is the sophisticated part that prevents cheating:

**Step 1 - Model Grouping:**
- Groups all results by AI model (not by miner)
- This collapses "Sybil attacks" where one person runs many copies of the same model
- If 10 miners run the same model, they're treated as one entity for performance calculation

**Step 2 - Domain-Specific GRPO Calculation:**
- For each domain (SAT, ABD, etc.), calculates each model's average performance
- Computes the group average across all models in that domain
- GRPO score = (model's average score) ÷ (group average score)
- This creates relative performance - a score of 1.2 means 20% better than average

**Step 3 - Multi-Domain Final Scoring:**
- Each model gets GRPO scores across multiple domains
- Final score = minimum GRPO score across all required domains
- This punishes models that are good at one thing but bad at others
- Encourages balanced, general-purpose AI rather than specialists

**Step 4 - Skew Penalty:**
- Adds a penalty for models with very uneven performance across domains
- If a model scores 2.0 in SAT but 0.5 in ABD, it gets penalized
- This further encourages balanced performance

**Step 5 - Winner Selection (Two-Step Process):**
- **First**: Identify the model with the highest final score across all models
- **Second**: Among all miners using the winning model, select the miner with the oldest block commitment
- **Example**: If Model A scores 1.4, Model B scores 1.2, Model C scores 0.8, then Model A wins. If miners [5, 12, 23] all use Model A with blocks [1000, 950, 1200], then Miner 12 wins (oldest commitment at block 950)
- Uses "winner-take-all" - only the oldest miner of the best model gets rewards
- Scores are smoothed over time using exponential moving averages to prevent volatility

## **4. The Complete Validation Cycle**

**Initialization:**
- Connects to Bittensor network and discovers active miners
- Filters to only miners that have registered working AI models
- Loads or generates challenge samples for each environment

**Testing Round:**
- Selects a set of challenges from each environment (e.g., 10 SAT + 10 ABD challenges)
- Sends all challenges to all available miners in parallel
- Collects responses with performance metrics (latency, success/failure)

**Evaluation:**
- Scores each individual response
- Groups results by model to collapse Sybil clones
- Calculates GRPO scores for each domain using aggregate performance from all miners per model
- Determines final scores for each model (min across domains with skew penalty)
- Identifies winner in two steps:
  1. Find the highest-scoring model
  2. Among miners using that model, select the one with oldest block commitment
- Updates historical performance records

**Continuous Operation:**
- Repeats the cycle with a configurable delay
- Maintains running averages of model performance
- Automatically generates new challenge samples in background
- Saves all results for historical analysis and leaderboards

## **5. Anti-Gaming Measures**

**Sybil Resistance:**
- Groups by model hash, not individual miners
- One model gets one score regardless of how many miners run it
- Two-step winner selection: best model first, then oldest miner within that model
- Prevents both Sybil attacks (multiple miners, same model) and model copying (late adopters stealing credit)

**Sample Management:**
- Each challenge is used only once and then deleted
- Continuous generation of fresh challenges prevents memorization
- Background stock replenishment ensures availability

**Multi-Domain Requirements:**
- Models must perform well across ALL required domains
- Can't specialize in just one area and ignore others

**Relative Scoring:**
- GRPO prevents absolute score inflation
- Performance is always relative to the current competition
- As the field improves, standards automatically rise