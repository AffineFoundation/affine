"""The GPU side of one box: LoRA training, multi-LoRA evaluation and the fold.

Everything that needs no GPU is plain Python and is tested on CPU:

  plan.py       elastic GPU split, the worker-image pin, the code digest
  batch.py      frozen batch -> training rows: caps, advantages, derail mask
  procs.py      child processes that hold GPUs, and the check that GPUs are free
  resident.py   verified local copies of the head, the fp32 master and LoRAs
  trainer.py    ``Trainer`` seam: one child process per LoRA, then hash and upload
  evaluator.py  ``Evaluator`` seam: vLLM servers, seeds, outcome codes
  folder.py     ``Folder`` seam: verify, merge, hash and upload
  genesis.py    one command: download the launch model, build its fp32 master

Only two modules import torch, and both stay thin: ``train_job.py`` (the
GRPO step, a child process) and ``folder.py`` (the fold arithmetic, on CPU).
vLLM is a separate process that the evaluator talks to over HTTP, so it is
never imported.
"""
