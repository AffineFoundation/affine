We need to reimplement the ENTIRE affine codebase. Delete everything. Rebuild everything. The core of the new codebase is to make everything principled from the start. We will still be a bittensor subnet. We will still use environments and evaluate them. But the way things are done is different.


1) Instead of the current environments, we will implement environments differently
* All environments are implemented using the openai/farama Gym interface
* All environments give a "challenge id" and other metadata for external verification
* All environments provide entirely procedurally generated content. Generation is deterministic and can be fully reproduced using the challenge id
* Initial environments are tic-tac-toe (for multi-turn) and 8-digit multiplication (for single-turn)

2) Instead of the current round-robin sampling and scoring, we will rewrite everything
* We only ever compare the "contender" with the current best miner. If the best miner is dethroned, the contender is the new point of reference
* We fully evaluate contender vs best until we have a certain result of who outperforms, based on the wilson score interval
* As there are multiple environments, we need to do aggregation globally
    * We first score each environment in parallel, comparing who won per challenge, until the win rate converged on one miner being confidently better
    * We continue sampling the environment until we know which miner is better
    * We will continue going through the environments with early stopping, until we know that the contender won on (N+K)/(2N) environments (i.e., if he won 0/N - we don't have to do the remaining N challenges)
    * If the contender wins, we increase the "ratio to beat" from "50%+eps with high confidence" (i.e., we know he's better, even if not by much) to the geometric mean of the ratios the winning contender got. So, a new miner would need to have the same geometric mean, or better, when beating this. This geometric mean will go down over time using exponential decay
    
3) Validators sample independently, set weights together
* Each validator samples the miners themselves and feeds the data into a shared bucket
* The samples contain the challenge id, chutes invocation id, full conversation transcript - and are uploaded in "blocks" of data - each block has a hash and refers to the previous block hash
* VTrust is assigned for how high the ratio of correct samples for a given miner is - if they're wrong, they lose vtrust
* Validators pull each other's data and set the weights (winner-takes-all) based on the samples of all validators

The separate validator buckets and commitments can be kept. However, AgentGym, Pareto Dominance, round robin sampling, and many other things have to be removed.

In the new implementation, aim for minimalist code. Don't add a line if it doesn't truly have to be there. The code has to be robust, so that miners and validators can't exploit it. However, it also has to be simple, minimal and easy to maintain. Most importantly, aim for a low number of lines and clean code. The co
