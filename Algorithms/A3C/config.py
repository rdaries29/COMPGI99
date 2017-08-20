# Course: COMPGI99 - Thesis
# Institution: University College London
# Developer: Russel Daries (16079408)

class config:

    GAME = 'Humanoid'
    TRAIN_MODELS = True
    LOAD_CHECKPOINT = False
    LOAD_EPISODE = 0

    USE_GPU = False
    USE_LSTM = True

    if USE_GPU:
        device = "/gpu:0"
    else:
        device = "/cpu:0"

    AGENTS = 32
    PREDICTOR = 2
    TRAINERS = 2

    DISCOUNT = 0.99

    REWARD_MIN = -1
    REWARD_MAX = 1

    MAX_QUEUE_SIZE = 4
    PREDICTION_BATCH_SIZE = 128

    EPISODES = 400000
    ANNEALING_EPISODE_COUNT = 400000

    BETA_START = 0.01
    BETA_END = 0.01

    LEARNING_RATE_START =
    LEARNING_RATE_END =

    USE_GRAD_CLIP = False
    GRAD_CLIP_NORM = 40.0

    TENSORBOARD = False
    TENSORBOARD_UPDATE_FREQ = 1000

    SAVE_MODELS = True
    SAVE_MODELS_FREQ = 1000

    PRINT_STATS_FREQ = 1


    RESULTS_FILENAME = 'results.txt'
    NETWORK_NAME = 'network'


    MIN_POLICY = 0.0
    USE_LOG_SOFTMAX = False