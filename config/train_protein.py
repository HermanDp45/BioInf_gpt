out_dir = 'out-protein'
eval_interval = 50
log_interval = 10
max_iters = 1000

# dir name
dataset = 'protein'

batch_size = 32
block_size = 155

n_layer = 6
n_head = 8
n_embd = 256
dropout = 0.2

learning_rate = 6e-4
device = 'cpu'
compile = False

class_prob = 0.5
type_prob = 0.3
data_type = 'init_seq'
prefix_mode = 'after_eos'

