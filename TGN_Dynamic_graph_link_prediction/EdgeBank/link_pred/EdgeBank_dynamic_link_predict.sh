data=wikipedia
mem_mode=unlim_mem
n_runs=5
neg_sample=rnd  # can be one of these options: "rnd": standard randome, "hist_nre": Historical NS, or "induc_nre": Inductive NS
python edge_bank_baseline.py --data "$data" --mem_mode "$mem_mode" --n_runs "$n_runs" --neg_sample "$neg_sample"