"""Runner-output -> envelope adapters. verifiers evals emit trace v1
natively; mini-swe-agent trajectories are converted into the same shape so
every downstream stage (store, index, views) is single-path."""
