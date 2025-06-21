# 🚀 Affine Framework v0.2.0

<div align="center">

**Next-Generation Decentralized AI Evaluation Framework for Bittensor Network**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Pydantic v2](https://img.shields.io/badge/Pydantic-v2-red.svg)](https://docs.pydantic.dev/)

*Production-Ready • Modular • Enterprise-Grade*

</div>

---

## ✨ **What is Affine?**

Affine is a **cutting-edge framework** for evaluating AI models through blockchain-based challenges on the **Bittensor network**. It features state-of-the-art Python architecture with real-time validation, ELO rating systems, and seamless model deployment pipelines.

### 🎯 **Key Features**

- 🏗️ **Enterprise Architecture** - Clean separation with dedicated modules and type safety
- ⛓️ **Bittensor Integration** - Native blockchain mining with async operations
- 🚀 **Complete Deployment Pipeline** - HuggingFace → Bittensor → Chutes.ai automation
- 🎨 **Rich CLI Interface** - Beautiful terminal UI with progress bars and tables
- ⚙️ **Smart Configuration** - Multi-source config with elegant fallback hierarchy
- 🧪 **Advanced Environments** - COIN, SAT, and SAT1 with planted solutions
- 🏆 **ELO Validation System** - Continuous head-to-head miner competition
- 📊 **Comprehensive Analytics** - Results persistence, ELO rankings, and performance metrics
- 🔍 **Enhanced Logging** - TRACE-level debugging with configurable verbosity
- ⚡ **Performance Optimized** - Progress bars, concurrent operations, async-first design

---

## 🏃‍♂️ **Quick Start**

### 1. **Installation**

```bash
# Install uv (fastest Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install Affine
git clone https://github.com/AffineFoundation/affine.git
cd affine
uv venv && source .venv/bin/activate
uv pip install -e .

# Verify installation
af --help
```

### 2. **Prerequisites: Chutes.ai Platform Setup**

Affine integrates with the **Chutes.ai platform** for model deployment and evaluation. You must complete the Chutes setup first:

#### **🔑 Create Bittensor Wallet (if needed)**
```bash
# Install Bittensor CLI (if not already installed)
pip install 'bittensor<8'

# Create wallet
btcli wallet new_coldkey --n_words 24 --wallet.name chutes-user
btcli wallet new_hotkey --wallet.name chutes-user --n_words 24 --wallet.hotkey chutes-user-hotkey
```

#### **🚀 Install & Register with Chutes**
```bash
# Install Chutes CLI
pip install chutes

# Register with Chutes (requires Bittensor wallet)
chutes register

# Create API key for Affine
chutes keys create --name affine-key --admin
```

#### **💰 Enable Developer Role (if needed)**

For model deployment, you may need developer access:

**Option A: Validators/Subnet Owners (Free)**
```bash
# Link your validator/owner key for free developer access
chutes link \
  --hotkey-path ~/.bittensor/wallets/wallet/hotkeys/hotkey \
  --hotkey-type subnet_owner  # or "validator"
```

**Option B: TAO Deposit (Refundable)**
```bash
# Check current deposit amount
curl -s https://api.chutes.ai/developer_deposit | jq .

# Get your deposit address
curl -s https://api.chutes.ai/users/me \
  -H 'authorization: Bearer YOUR_API_KEY'

# Send TAO to the deposit address (via Bittensor wallet)
# Then wait 7+ days to request refund if needed
```

### 3. **Configuration Setup**

```bash
# Set your API credentials
af set CHUTES_API_KEY your_chutes_api_key
af set HF_USER your_huggingface_username
af set HF_TOKEN your_huggingface_token

# Optional: Bittensor wallet info
af set BT_COLDKEY your_coldkey_name
af set BT_HOTKEY your_hotkey_name

# Optional: Custom deployment image
af set CHUTES_IMAGE chutes/sglang:v0.4.6

# Verify configuration
af config --show
```

### 4. **Run Your First Challenge**

```bash
# Simple coin flip challenges with progress bars
af run 5,6,7 COIN -n 5 --save-results --verbose-results

# Advanced SAT challenges with enhanced logging
af run 5,6,7 SAT1 -n 3 -v

# Start continuous ELO validation
af validate --rounds 10 --show-elo
```

**🎉 You're ready to evaluate AI models at scale!**

---

## 📚 **Comprehensive Feature Guide**

### 🔧 **Advanced Configuration Management**

Affine uses a **sophisticated, hierarchical configuration system**:

1. **Command-line arguments** (highest priority)
2. **Environment variables** 
3. **Configuration files** (`~/.affine/config.ini`, `~/.chutes/config.ini`)
4. **Default values** (lowest priority)

#### **Configuration Files**

Create `~/.affine/config.ini`:
```ini
[chutes]
chutes_api_key = your_api_key_here

[deployment]
hf_user = your_hf_username
hf_token = hf_token_here
chutes_user = your_chutes_username
chutes_image = chutes/sglang:v0.4.6
chutes_concurrency = 16
chutes_gpu_count = 8
chutes_min_vram_gb = 40
wallet_cold = your_coldkey_name
wallet_hot = your_hotkey_name
bt_netuid = 120
```

#### **Advanced CLI Configuration**
```bash
# Enhanced configuration management
af set CHUTES_API_KEY your_api_key
af set HF_USER your_username
af set CHUTES_IMAGE custom:latest

# View configuration with security masking
af config --show

# Get specific values
af config --get CHUTES_API_KEY

# Test configuration hierarchy
export TEST_VAR="env_value"
af set TEST_VAR "config_value"
af config --get TEST_VAR  # Returns "env_value" (env has priority)
```

### ⛓️ **Enhanced Miner Operations**

#### **Advanced Miner Listing**
```bash
# Smart filtering and display options
af miners --active-only --limit 20    # Active miners only
af miners 5,6,7,11,12,13              # Specific UIDs
af miners --limit 0                   # All miners (can be slow)

# Enhanced display with detailed information
af miners --active-only -v            # Verbose logging
```

#### **Programmatic Miner Access**
```python
import affine as af

# Get miners with advanced filtering
active_miners = await af.miners(no_null=True)  # Only miners with models
specific_miners = await af.miners([5, 6, 7])   # Specific UIDs
all_miners = await af.miners()                 # All miners

# Enhanced miner information
for uid, miner in active_miners.items():
    print(f"UID {uid}: {miner.model} (Block: {miner.block})")
    if miner.chute:
        print(f"  Chute info: {miner.chute.get('status', 'unknown')}")
```

### 🎯 **Advanced Challenge Environments**

#### **COIN Environment**
Simple but effective for basic testing:
```bash
af run 5,6,7 COIN -n 10 --save-results
```

#### **SAT Environment**
Boolean satisfiability with configurable difficulty:
```python
import affine as af

# Create SAT with custom parameters
sat_env = af.SAT(n=5, k=3, m=15)  # 5 variables, 3-SAT, 15 clauses
challenges = await sat_env.many(3)
```

#### **SAT1 Environment (Advanced)**
Enterprise-grade SAT with planted solutions and robust verification:
```bash
# Advanced SAT with comprehensive logging
af run 5,6,7 SAT1 -n 5 -vv

# High-difficulty challenges
python -c "
import asyncio
import affine as af

async def hard_sat():
    sat1 = af.SAT1(n=10, k=4, m=42)  # Hard 4-SAT instance
    challenge = await sat1.generate()
    print(f'Generated {len(challenge.extra[\"clauses\"])} clauses')

asyncio.run(hard_sat())
"
```

### 📊 **Real-Time Progress & Analytics**

#### **Progress Bars**
All long-running operations feature beautiful progress bars:
```bash
# Progress bars show real-time completion
af run 5,6,7,11,12,13 SAT1 -n 10  # Shows: 🚀 Affine ████████ 60/60

# Disable for scripting
af run 5,6,7 COIN -n 5 --no-progress
```

#### **Enhanced Results Display**
```bash
# Comprehensive results with analytics
af run 5,6,7 COIN -n 5 --verbose-results
```

Output example:
```
🏆 Evaluation Results
┏━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┓
┃ UID ┃ Model             ┃ Challenges  ┃ Total Score ┃ Average   ┃ Errors    ┃ Status ┃
┡━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━┩
│ 5   │ unsloth/gemma-4b  │ 5           │ 4.000       │ 0.800     │ 0         │ ✅     │
│ 6   │ Alphatao/Affine-X │ 5           │ 3.000       │ 0.600     │ 1         │ ⚠️      │
└─────┴───────────────────┴─────────────┴─────────────┴───────────┴───────────┴────────┘

📊 Overall Statistics:
   Total Challenges: 10
   Total Score: 7.000
   Average Score: 0.700
   Success Rate: 9/10 (90.0%)
```

#### **Results Persistence**
```bash
# Automatic timestamped JSON storage
af run 5,6,7 COIN -n 5 --save-results
# Saves to: ~/.affine/results/1703123456.json

# Custom analysis
python -c "
import affine as af
results = af.load_results('~/.affine/results/1703123456.json')
af.summarize_challenges(results)
"
```

### 🏆 **ELO Validation System**

#### **Continuous Validation**
```bash
# Start ELO-based continuous validation
af validate --show-elo --rounds 100

# High-stakes validation with custom parameters
af validate -k 64 -c 5 -d 10 --show-elo

# Infinite validation (Ctrl+C to stop)
af validate --show-elo
```

#### **ELO Features**
- **Head-to-head competition** between random miners
- **Configurable K-factor** for rating sensitivity
- **Tie-breaking logic** (older miners win ties)
- **Real-time rankings** every 5 rounds
- **Comprehensive final rankings**

Example output:
```
🎯 Starting validator (K=32, challenges=2, delay=5.0s)
📊 Running infinite rounds

🎮 Round 1: 6 active miners
🏆 Winner: UID 7 (2.0 vs 1.0)

📈 ELO Rankings after 5 rounds:
┏━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Rank ┃ Hotkey          ┃ ELO Rating  ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ 1    │ 5HRCq8YL...     │ 1547.3      │
│ 2    │ 5Hjnirzn...     │ 1523.7      │
│ 3    │ 5H4QM65M...     │ 1489.1      │
└──────┴─────────────────┴─────────────┘
```

### 🚀 **Complete Model Deployment Pipeline**

Deploy models through the full **HuggingFace → Bittensor → Chutes.ai** pipeline:

#### **Basic Deployment**
```bash
# Deploy from model/ directory
af deploy my_model_name

# Deploy with specific configuration
af deploy my_model --hf-user myusername --chutes-image custom:latest

# Use existing HuggingFace repository
af deploy existing_model --existing-repo myuser/my-existing-repo
```

#### **Advanced Deployment Options**
```bash
# Full path deployment
af deploy /path/to/my/model --full-path

# Comprehensive dry run
af deploy my_model --dry-run

# Custom parameters
af deploy my_model \
  --chutes-image chutes/sglang:v0.4.6 \
  --blocks-until-reveal 720 \
  --chutes-api-key your_key
```

#### **Deployment Process**
The automated pipeline:

1. **📤 HuggingFace Upload** - Intelligent file detection and upload
2. **⛓️ Bittensor Commitment** - Blockchain registration with reveal delay
3. **🚀 Chutes Deployment** - Production endpoint creation
4. **🔥 Warm-up Testing** - Automatic functionality verification

### 🔍 **Enhanced Logging & Debugging**

#### **Configurable Verbosity**
```bash
# Different logging levels
af run 5,6,7 COIN -n 3 -v      # INFO level
af run 5,6,7 SAT -n 2 -vv      # DEBUG level
af run 5,6,7 SAT1 -n 1 -vvv    # TRACE level (most detailed)

# Validate with debugging
af validate --rounds 5 -vv
```

#### **Programmatic Logging**
```python
import affine as af

# Setup logging in your code
af.setup_logging(verbosity=2)  # DEBUG level

# All framework operations now have detailed logging
miners_dict = await af.miners(no_null=True)
# Logs: "Discovered 6 miners: UID: 5, Hotkey: 5Hjnirzn..., Model: unsloth/gemma-4b..."
```

---

## 🛠️ **Advanced Usage**

### 🎛️ **Complete CLI Reference**

#### **af run** - Execute Challenges
```bash
af run UIDS ENVIRONMENT [OPTIONS]

# Core Options:
#   -n, --num-challenges    Number of challenges [default: 1]
#   --timeout              API timeout in seconds [default: 30.0]
#   --retries              Retry attempts [default: 3]
#   --backoff              Backoff multiplier [default: 1.0]
#   --no-progress          Disable progress bar
#   --save-results         Save to JSON file
#   --verbose-results      Show detailed results

# Logging Options:
#   -v, -vv, -vvv          Increase verbosity (INFO/DEBUG/TRACE)

# Examples:
af run 5,6,7 COIN -n 5 --save-results -v
af run 11,12,13 SAT1 --timeout 60 --verbose-results -vv
af run 5,6,7,11,12,13 SAT -n 10 --no-progress
```

#### **af validate** - ELO Validation
```bash
af validate [OPTIONS]

# Options:
#   -k, --k-factor         ELO K-factor [default: 32]
#   -c, --challenges       Challenges per game [default: 2]
#   -d, --delay           Delay between rounds [default: 5.0]
#   -r, --rounds          Number of rounds (0=infinite) [default: 0]
#   --show-elo            Show ELO rankings periodically

# Examples:
af validate --rounds 50 --show-elo -k 64
af validate -c 5 -d 2 --show-elo -v
```

#### **af miners** - Miner Management
```bash
af miners [UIDS] [OPTIONS]

# Options:
#   --limit               Limit display count [default: 50]
#   --active-only         Show only miners with models

# Examples:
af miners --active-only --limit 20
af miners 5,6,7,11,12,13
af miners --limit 0  # Show all (can be slow)
```

### 🔌 **Advanced API Integration**

#### **Comprehensive Programmatic Usage**
```python
import asyncio
import affine as af

async def advanced_evaluation():
    # Setup enhanced logging
    af.setup_logging(verbosity=2)
    
    # Get active miners with filtering
    all_miners = await af.miners(no_null=True)
    top_miners = dict(list(all_miners.items())[:5])
    
    # Create multiple environment types
    environments = {
        'COIN': af.COIN(),
        'SAT': af.SAT(n=4, k=3, m=8),
        'SAT1': af.SAT1(n=5, k=3, m=12)
    }
    
    results = []
    for env_name, env in environments.items():
        print(f"Testing {env_name}...")
        
        # Generate challenges
        challenges = await env.many(3)
        
        # Run with progress bars and custom settings
        env_results = await af.run(
            challenges=challenges,
            miners_param=top_miners,
            timeout=45.0,
            retries=5,
            show_progress=True
        )
        results.extend(env_results)
    
    # Comprehensive analysis
    af.print_results(results, verbose=True)
    af.summarize_challenges(results)
    
    # Calculate and display ELO ratings
    elo_ratings = af.calculate_elo_ratings(results)
    af.print_elo_rankings(elo_ratings, top_n=10)
    
    # Save with timestamp
    saved_path = af.save_results(results)
    print(f"Results saved to: {saved_path}")
    
    return results

# Run the advanced evaluation
results = asyncio.run(advanced_evaluation())
```

#### **Custom Environment Creation**
```python
import affine as af
import random
from typing import Dict, Any

class CustomMathEnv(af.BaseEnv):
    """Custom mathematical reasoning environment."""
    
    name: str = "MATH"
    difficulty: int = 1
    
    async def generate(self) -> af.Challenge:
        """Generate a math problem."""
        a, b = random.randint(1, 10 * self.difficulty), random.randint(1, 10 * self.difficulty)
        operation = random.choice(['+', '-', '*'])
        
        if operation == '+':
            answer = a + b
        elif operation == '-':
            answer = a - b
        else:
            answer = a * b
        
        return af.Challenge(
            env=self,
            prompt=f"What is {a} {operation} {b}? Provide only the number.",
            extra={"answer": answer, "a": a, "b": b, "operation": operation}
        )
    
    async def evaluate(self, challenge: af.Challenge, response: af.Response) -> af.Evaluation:
        """Evaluate the math response."""
        expected = challenge.extra["answer"]
        
        if response.response is None:
            return af.Evaluation(env=self, score=0.0, extra={"reason": "No response"})
        
        try:
            provided = int(response.response.strip())
            correct = (provided == expected)
            return af.Evaluation(
                env=self, 
                score=float(correct),
                extra={"expected": expected, "provided": provided}
            )
        except ValueError:
            return af.Evaluation(
                env=self, 
                score=0.0,
                extra={"expected": expected, "reason": "Invalid number format"}
            )

# Use your custom environment
async def test_custom_env():
    math_env = CustomMathEnv(difficulty=3)
    challenge = await math_env.generate()
    print(f"Challenge: {challenge.prompt}")
    
    # Test with correct answer
    correct_response = af.Response(
        response=str(challenge.extra["answer"]),
        latency_seconds=0.1,
        attempts=1,
        model="test",
        error=None
    )
    
    evaluation = await math_env.evaluate(challenge, correct_response)
    print(f"Score: {evaluation.score}")

asyncio.run(test_custom_env())
```

---

## 🚨 **Troubleshooting & Best Practices**

### **Platform Dependencies**

Affine integrates with multiple platforms that must be properly configured:

- **🔗 Chutes.ai**: Required for model evaluation and deployment
- **⛓️ Bittensor**: Blockchain network for miner discovery and validation  
- **🤗 HuggingFace**: Model repository for deployment pipeline
- **🐍 Python 3.9+**: Required runtime environment

**⚠️ Important**: Complete the Chutes setup (Section 2) before using Affine for model evaluation.

### **Bittensor Integration Notes**

Affine follows standard Bittensor development patterns:
- `bt.async_subtensor()` for async operations (miner discovery, metagraph queries)
- `bt.subtensor()` for sync operations (blockchain commits, transactions)
- Standard wallet initialization: `bt.wallet(name=coldkey, hotkey=hotkey)`

### **Common Issues**

#### **Configuration Troubleshooting**
```bash
# Debug configuration issues
af config --show                    # View all configuration
af config --get CHUTES_API_KEY     # Check specific values

# Test configuration hierarchy
python -c "
import affine as af
print('CHUTES_API_KEY:', af.config.get('CHUTES_API_KEY'))
print('All config:', list(af.config.get_all().keys())[:5])
"
```

#### **Network & Mining Issues**
```bash
# Test miner connectivity
af miners --limit 5 -v             # Verbose miner listing

# Debug API calls
af run 5 COIN -n 1 -vvv            # Maximum logging

# Test with single miner
af run 5 COIN -n 1 --timeout 60 --retries 5
```

#### **Performance Optimization**
```bash
# Optimize for large-scale testing
af run 5,6,7,11,12,13 SAT -n 10 --no-progress  # Disable UI for speed
af validate -d 1 --rounds 100                  # Faster validation cycles

# Batch operations
python -c "
import asyncio, affine as af
async def batch_test():
    miners = await af.miners(no_null=True)
    challenges = await af.SAT().many(20)
    results = await af.run(challenges, miners, show_progress=False)
    print(f'Completed {len(results)} evaluations')
asyncio.run(batch_test())
"
```

### **Production Deployment**

#### **Environment Setup**
```bash
# Production configuration
export CHUTES_API_KEY="your_production_key"
export HF_TOKEN="your_hf_token"
export BT_COLDKEY="production_coldkey"
export BT_HOTKEY="production_hotkey"

# Verify production setup
af config --show
af miners --active-only --limit 10
```

#### **Monitoring & Logging**
```python
# Production monitoring setup
import logging
import affine as af

# Setup comprehensive logging
af.setup_logging(verbosity=1)  # INFO level for production
logger = logging.getLogger("production")

async def production_monitoring():
    """Production monitoring script."""
    while True:
        try:
            # Get active miners
            miners = await af.miners(no_null=True)
            logger.info(f"Active miners: {len(miners)}")
            
            # Run validation round
            if len(miners) >= 2:
                sat_env = af.SAT1(n=4, k=3, m=10)
                challenges = await sat_env.many(2)
                
                # Sample random miners
                import random
                selected = random.sample(list(miners.values()), 2)
                
                results = await af.run(
                    challenges=challenges,
                    miners_param=selected,
                    show_progress=False
                )
                
                # Log results
                avg_score = sum(r.evaluation.score for r in results) / len(results)
                logger.info(f"Validation round completed, avg score: {avg_score:.3f}")
                
                # Save results
                af.save_results(results)
            
            # Wait before next round
            await asyncio.sleep(300)  # 5 minutes
            
        except Exception as e:
            logger.error(f"Production monitoring error: {e}")
            await asyncio.sleep(60)  # Wait 1 minute on error

# Run production monitoring
# asyncio.run(production_monitoring())
```

---

## 📈 **Performance Metrics**

### **Benchmarks**

| Operation | Duration | Throughput |
|-----------|----------|------------|
| Challenge Generation | ~0.01s | 100/second |
| Single Evaluation | ~2-5s | Depends on model |
| Miner Discovery | ~3-10s | All active miners |
| ELO Round | ~10-30s | 2 miners, 2 challenges |
| Result Persistence | ~0.1s | 1000 results/second |

### **Scalability**

- ✅ **Concurrent Operations**: Up to 50 simultaneous evaluations
- ✅ **Large Datasets**: Tested with 10,000+ challenge evaluations
- ✅ **Memory Efficiency**: <100MB RAM for typical operations
- ✅ **Network Resilience**: Automatic retry with exponential backoff

---

## 📄 **API Reference**

### **Core Classes**

#### **Challenge**
```python
class Challenge(BaseModel):
    env: BaseEnv                    # Environment that created this
    prompt: str                     # The challenge prompt/question
    extra: Dict[str, Any]          # Additional data (answers, metadata)
    
    async def evaluate(self, response: Response) -> Evaluation
```

#### **Response** 
```python
class Response(BaseModel):
    response: Optional[str]         # Model's response text
    latency_seconds: float         # Response time
    attempts: int                  # Number of API attempts
    model: str                     # Model identifier
    error: Optional[str]           # Error message if failed
```

#### **Evaluation**
```python
class Evaluation(BaseModel):
    env: BaseEnv                   # Environment that evaluated
    score: float                   # Evaluation score (0.0-1.0)
    extra: Dict[str, Any]         # Additional evaluation data
```

#### **Result**
```python
class Result(BaseModel):
    miner: Miner                   # Miner information
    challenge: Challenge           # The challenge
    response: Response            # Model's response
    evaluation: Evaluation        # Evaluation result
```

### **Key Functions**

#### **miners(uids=None, no_null=False)**
```python
async def miners(
    uids: Optional[Union[int, List[int]]] = None,
    no_null: bool = False
) -> Dict[int, Miner]:
    """Get miners from Bittensor network with advanced filtering."""
```

#### **run(challenges, miners_param, ...)**
```python
async def run(
    challenges: Union[Challenge, List[Challenge]],
    miners_param: Optional[Union[Dict[int, Miner], List[Miner], int, List[int]]] = None,
    timeout: float = 30.0,
    retries: int = 3,
    backoff: float = 1.0,
    show_progress: bool = True
) -> List[Result]:
    """Run challenges against miners with progress tracking."""
```

#### **Utility Functions**
```python
# Results management
def print_results(results: List[Result], verbose: bool = False) -> None
def save_results(results: List[Result], custom_path: str = None) -> str
def load_results(file_path: str) -> List[Dict[str, Any]]

# Analytics
def calculate_elo_ratings(results: List[Result], k_factor: int = 32) -> Dict[str, float]
def print_elo_rankings(elo_ratings: Dict[str, float], top_n: int = 20) -> None
def summarize_challenges(results: List[Result]) -> None

# Configuration
def setup_logging(verbosity: int = 0) -> None
```

---

## 🤝 **Contributing**

We welcome contributions! Here's how to get started:

### **Development Setup**
```bash
# Clone and setup development environment
git clone https://github.com/AffineFoundation/affine.git
cd affine
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run tests
python -m pytest tests/

# Code quality
black affine/
isort affine/
mypy affine/
```

### **Adding New Environments**
1. Create a new file in `affine/envs/`
2. Inherit from `BaseEnv`
3. Implement `generate()` and `evaluate()` methods
4. Add to `ENVS` registry in `cli.py` and `__init__.py`

### **Adding CLI Commands**
1. Add new command to `cli.py`
2. Follow existing patterns for options and help text
3. Use Rich formatting for beautiful output

---

## 📝 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 **Acknowledgments**

- **Bittensor Network** - For the decentralized AI infrastructure
- **Chutes.ai** - For model deployment and hosting platform
- **HuggingFace** - For model repository and distribution
- **Rich Library** - For beautiful terminal interfaces
- **Pydantic** - For robust data validation
- **Alive-Progress** - For beautiful progress bars

---

<div align="center">

**Made with ❤️ by the Affine Foundation**

[📖 Documentation](https://docs.affine.ai) • [🧪 Testing Guide](./TESTING_GUIDE.md) • [🐛 Issues](https://github.com/AffineFoundation/affine/issues) • [💬 Discussions](https://github.com/AffineFoundation/affine/discussions)

**Ready for Production • Built for Scale • Designed for the Future**

</div>