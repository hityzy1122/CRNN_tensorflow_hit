

GPU_ID = 0
PATH_DICT_TXT = './data/chinese_dict.txt'
PATH_STR_2_LABEL_DICT = './data/dict/str_2_label.json'
PATH_LABEL_2_STR_DICT = './data/dict/label_2_str.json'
PATH_CHECKPOINTS = './checkpoints'
PATH_IMAGES = 'data/images'
PATH_TFRECORDS_TRAIN = './data/tfrecords/train.tfrecords'
PATH_TFRECORDS_VALIDATION = './data/tfrecords/val.tfrecords'
PATH_TRAIN_LIST = './data/sample_train.txt'
PATH_VAL_LIST = './data/sample_val.txt'
PATH_TEST_IMAGES = './data/test/'

IMAGE_WIDTH = 100
IMAGE_HEIGHT = 32

TRAIN_BATCH_SIZE = 128
TRAIN_VAL_BATCH_SIZE = 128
TRAIN_LEARNING_RATE = 0.00001
TRAIN_CLASS_NUMS = 3786+1
TRAIN_LOAD_MODEL = None
TRAIN_SEQUENCE_LENGTH = 25
TRAIN_EPHO = 100000
TRAIN_DISPLAY = 1
TRAIN_VAL_SNAPSHOT = 1000

TEST_CHECKPOINTS = './checkpoints/20180625-2107/'
