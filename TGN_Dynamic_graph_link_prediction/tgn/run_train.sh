data=wikipedia
n_runs=5

# TGN
method=tgn
prefix="${method}_attn"
python train_self_supervised.py -d $data --use_memory --prefix "$prefix" --n_runs "$n_runs" --gpu 0

# method:tgn   data:wikipedia    prefix:tgn_attn   n_runs:5