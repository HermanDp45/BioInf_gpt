# выходная папка для модели и чекпоинтов
out_dir = 'out-protein'

# оценка на валидации
eval_interval = 50
# интервал логирования
log_interval = 10
# макс. итераций тренировки
max_iters = 1000

# папка с данными (метафайл meta.pkl)
dataset = 'protein'

# параметры датасета
# размер батча
batch_size = 32
# макс. длина последовательности в батчe
block_size = 155

# количество слоев, голов внимания и размер эмбеддинга
n_layer = 6
n_head = 8
n_embd = 256
# дроп-аут
dropout = 0.2
# скорость обучения
learning_rate = 6e-4
# устройство для тренировки
device = 'cpu'
# компиляция модели
compile = False

# вероятности для добавления class и type токенов
class_prob = 0.5
type_prob = 0.3
# тип последовательности
data_type = 'init_seq'
prefix_mode = 'after_eos'

