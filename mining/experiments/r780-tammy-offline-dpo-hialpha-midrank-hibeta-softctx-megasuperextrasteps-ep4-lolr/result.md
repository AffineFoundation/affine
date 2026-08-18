p3848: MERGE_DONE → WAIT_RELAY armed (host_relay_r780_parallel_after_r768_p3848.sh pid 3405444).
p3858: meta accel cfg/tokenizer/vis-meta DONE on R252.
p3861: parallel×4 mid-flight (9/16 ok); killed stalled shard-11 ssh PIDs; launched tail accel 13–16+vis (host_relay_r780_tail_accel_p3861.sh pid 3606286). Waiter armed for SCP_READY→lean n80 :8002.
p3862: found dual-write on shard13 (p3848+tail) + wasteful 08 redo; SIGSTOP p3848 parent 3405444; killed 08/13 writers by PID; EOF-kill partial-mv corrupted 08 → size-safe 08fix pid 3617235; clean 13+vis pid 3616306; tail keeps 14/15/16; kids 11/12 still live under STOP'd parent.
p3863: 08+13 finished size-ok (16+vis verify); premature SCP_READY@12:05 (count-only) launched lean against truncated 08 → vLLM fail@~41%; killed STOP parent 3405444 (unblocks R781); relaunch lean pid463888 chall:8002 pid464063 → CHALL_READY + n80 pid467342 LIVE (~25/80 @12:14Z) vs reign35 wvk7.
