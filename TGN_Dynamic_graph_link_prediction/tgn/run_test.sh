n_runs=5
data=wikipedia
neg_sample=hist_nre  # can be either "hist_nre" for historical NS, or "induc_nre" for inductive NS
# TGN
method=tgn
python tgn_test_trained_model_self_sup.py -d $data --use_memory --model $method --neg_sample $neg_sample --n_runs $n_runs --gpu 0