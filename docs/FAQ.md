# Frequently Asked Questions (FAQ)

## Getting Started & General Questions

**Q1: What is the purpose of the Affine subnet?**

A: Affine is a Bittensor subnet designed to incentivize the creation of advanced reasoning models. The goal is to push the state-of-the-art (SOTA) in Reinforcement Learning (RL) and drive the development of more intelligent models by rewarding miners for improving performance on a variety of challenging tasks.

**Q2: How does a miner work on this subnet?**

A: Miners on Affine do not run mining hardware directly. Instead, the process is:
1. **Train a Model:** You find or train a machine learning model to perform well on the subnet's environments.
2. **Upload to Hugging Face:** You upload your trained model weights to a Hugging Face repository.
3. **Deploy on Chutes:** You deploy your model as a "Chute," which is a serverless inference endpoint. Affine's validators then send tasks to your Chute to evaluate your model's performance.
4. **Commit to Affine:** You commit your model's information (Chute ID, Hugging Face revision) to the Bittensor blockchain on subnet 120.

**Q3: What are the requirements to start mining?**

A: You will need:
* A Hugging Face account to host your models.
* A Chutes.ai account. You must register this account using the **same hotkey** you use to mine on Affine. This removes the need for a developer deposit.
* Funds in your Chutes account to pay for the GPU hours your model uses when it's active.

**Q4: Where can I find the official code and leaderboard?**

* **GitHub Repository:** [https://github.com/AffineFoundation/affine](https://github.com/AffineFoundation/affine)
* **Live Dashboard:** [https://www.affine.io/](https://www.affine.io/)

---

## Mining, Models, and Environments

**Q5: What kind of tasks (environments) does the subnet use for evaluation?**

A: The environments are under active development and change frequently. The subnet has moved from simpler environments (like SAT, ABD, DED) to more complex, multi-turn tasks from the **AgentGym** suite, including `webshop`, `alfworld`, `babyai`, and `sciworld`. The goal is to use challenging benchmarks where models cannot easily achieve 100% accuracy out of the box.

**Q6: What is the "model copying" problem everyone talks about?**

A: Model copying is a major issue where some miners download a successful model from another miner, make a trivial change to alter its hash, and redeploy it as their own. Due to statistical variance in scoring, these copies can sometimes outperform the original and "steal" emissions without contributing any new training or improvement.

**Q7: How is the team addressing the model copying exploit?**

A: The team has implemented and is continuing to develop several measures:
* **Statistical Significance:** The scoring algorithm was updated to use Beta distribution confidence intervals. A new model must show a statistically significant improvement over an existing one to be considered better, rather than winning due to random variance.
* **First Commit Advantage:** The system tracks the block number of a model's first submission. In the case of a tie, the earlier submission wins.
* **Future Plans:** The team is working on solutions like private evaluation windows, more advanced model fingerprinting (analyzing responses and internal states), and leveraging new security features from Chutes (like TEEs) to make copying ineffective.

---

## Troubleshooting and Technical FAQ

**Q8: I'm getting an error running the `af weights` command. How do I fix it?**

A: This is a common issue. Try these steps:
1. **Update Your Repo:** Make sure you have the latest code: `git pull`.
2. **Re-install Dependencies:** The requirements can change: `uv pip install -e .`.
3. **Check Environment Variables:** Ensure you have a `.env` file with the correct, up-to-date values copied from the `.env.example` file in the repository.
4. If it still fails with errors like `Unclosed client session` or `ValueError: max() arg is an empty sequence`, the data endpoints may be temporarily down for maintenance.

**Q9: My Chute is "cold" and I'm not getting any requests from validators. What's wrong?**

A: This happens for a few reasons:
* **Automatic Shutdown:** Chutes are designed to shut down if they are inactive for a period (default is 5-10 minutes) to save costs.
* **Validator Sampling:** Validators only check for "hot" (active) miners periodically. If your Chute is cold when they check, you won't get any tasks.

**How to fix it:**
1. **Increase Shutdown Time:** In your Chute configuration file, increase the `shutdown_after_seconds` parameter (e.g., to `1800` for 30 minutes) to keep your instance alive longer.
2. **Keep it Warm:** You can write a simple script to send a request to your own Chute every few minutes to prevent it from going cold.

**Q10: My Chute fails to deploy or won't become active. What should I do?**

A: This is almost always an issue with your Chute configuration.
* **Check the Logs!** This is the most critical step. A pinned message in the Discord provides a detailed guide on how to retrieve live logs from your Chute instance using its Instance ID and your Chutes API key.
* **Common Errors:**
  * **Invalid `engine_args`:** Ensure there are no typos or commas between arguments.
  * **Outdated Image:** The error `Must use image="chutes/sglang:YYYYMMDDHH" (or more recent...)` means you must update the `image` parameter in your Chute configuration to the specified version or a newer one.
  * **Corrupted Model:** Your `model.safetensors` file might be corrupted or uploaded incorrectly to Hugging Face.

**Q11: My new model has been submitted but doesn't appear on the leaderboard. Why?**

A: It can take time. The system evaluates models over a large window of blocks (e.g., 10,000 blocks, which can be over a day). If your model still doesn't appear, double-check that your on-chain commit was successful and that there isn't a mismatch between the model revision you committed and the one deployed on your Chute.

---

## Subnet Mechanics

**Q12: Why are emissions often at 10% or burning completely (0% to miners)?**

A: The team deliberately reduces or "burns" emissions by sending them to UID 0 when there are critical issues. This is done to ensure fairness and prevent exploiters from profiting. Reasons for burning have included:
* Fixing major security exploits (e.g., a Chutes vulnerability that allowed routing requests to GPT-4o).
* Correcting bugs in the scoring mechanism.
* Rolling out significant infrastructure changes that require validators to update.
* General network instability or downtime.

**Q13: How does the scoring and weight system work? What are the `L1` to `L8` columns?**

A: Affine uses a **Pareto dominance** scoring system. Instead of one model being the "winner-takes-all," the system identifies winners across all possible subsets of the available environments.
* The `L1` through `L8` columns show the points a miner has earned for being dominant on subsets of that corresponding size. For example, `L2` shows points for winning on two-environment combinations, while `L8` is for winning across all eight environments simultaneously.
* This encourages miners to develop models that are specialized and perform well on different combinations of tasks, not just a single generalist model.
