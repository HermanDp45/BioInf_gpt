# tiny model for CPU debugging
out_dir = 'out-tiny'
eval_interval = 10
log_interval = 1
max_iters = 50

dataset = 'protein'
batch_size = 8
block_size = 64

n_layer = 4
n_head = 4 
n_embd = 128
dropout = 0.1

learning_rate = 1e-3
device = 'cpu'
compile = False