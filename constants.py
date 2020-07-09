# ENVIRONMENT PARAMETERS
win_width = 800
win_height = 800
win_dims = (win_width, win_height)

cell_size = 50

vision_x = 4
vision_y = 4

# LEARNING PARAMETERS
replay_buffer_len = 1000000
batch_size = 50

inp_size = (vision_x * 2 + 1) * (vision_y * 2 + 1)
hidden_1 = 512
hidden_1_activation = 'relu'
hidden_2 = 512
hidden_2_activation = 'relu'
output = 4

learning_rate = 1e-3

gamma = 0.95

epsilon = 1
epsilon_dec = 0.999

rb_file = 'replay_buffer.pickle'
model_file = 'model.h5'

usage = \
'''
Self Learning Vacuum Cleaner
Usage: python <file> <mode>
Where <mode> can be:
    n - Train a new model
    c - Continue training an existing model
    e - Simply evaluate an existing model
'''
