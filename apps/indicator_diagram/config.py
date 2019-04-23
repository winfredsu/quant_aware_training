# dataset related params
IMG_SHAPE = [160, 160, 3]
DATASET_NAME = 'indicator_diagram'
DATASET_SIZE = 439
DATASET_SPLIT = [0.7, 0.2, 0.1] # TRAIN, VAL, TEST
NUM_CLASSES  = 7
BATCH_SIZE   = 1

TRAIN_SIZE = int(DATASET_SIZE*DATASET_SPLIT[0])
VAL_SIZE   = int(DATASET_SIZE*DATASET_SPLIT[1])
TEST_SIZE  = int(DATASET_SIZE*DATASET_SPLIT[2])
TRAIN_SPLIT = TRAIN_SIZE
VAL_SPLIT   = TRAIN_SIZE+VAL_SIZE
TEST_SPLIT  = DATASET_SIZE

# model related params
DEPTH_MULTIPLIER = 0.5
DROPOUT_PROB = 0.5
DATA_DIR = './dataset/'
