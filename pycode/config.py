from yacs.config import CfgNode as CN

_C = CN()

_C.POSE2IMAGE = True
_C.GOAL_INPUT = False # TODO

_C.STEP = 1 # 1 or 2
_C.PRED_LEN = 3
_C.IMAGE_PRED_LEN = 2 # invalid

###############################
# BASIC #######################
###############################
_C.BASIC = CN()
_C.BASIC.DEVICE = 'cuda'
_C.BASIC.OUTPUT_DIR = 'output'
_C.BASIC.SEED = 123
_C.BASIC.MAX_EPOCH = 1000
_C.BASIC.BATCH_SIZE = 8
_C.BASIC.NUM_GPU = 4
_C.BASIC.WORKERS = 16

###############################
# DATASET #####################
###############################

_C.DATASET = CN()
_C.DATASET.NAME = 'HMD' # HMD or RLBench 
_C.DATASET.RGB_AUGMENTATION = True
_C.DATASET.DEPTH_AUGMENTATION = False

# HMD
_C.DATASET.HMD = CN()

_C.DATASET.HMD.ACTION_CLASS = 'normal' #'normal', 'summary'

_C.DATASET.HMD.FRAME_INTERVAL = 3
_C.DATASET.HMD.RANDOM_LEN = 2  # random range -> FRAME_INTERVAL * RANDOM_LEN  

###############################
# AUGMENTATION ################
###############################

_C.AUGMENTATION = CN()

# image augmentation
_C.AUGMENTATION.FLIP_LR = 0.5   # not used
_C.AUGMENTATION.FLIP_UD = 0.5   # not used

_C.AUGMENTATION.GAUSSIAN_NOISE = 0.01 # old 0.05
_C.AUGMENTATION.GAUSSIAN_BLUR = 0.5 # old 1.5

_C.AUGMENTATION.CHANGE_BRIGHTNESS = False
_C.AUGMENTATION.BRIGHTNESS_MUL_RANGE = (0.8,1.1)
_C.AUGMENTATION.BRIGHTNESS_ADD_RANGE = (-30,10)

_C.AUGMENTATION.GAMMA_CONTRAST = 1.

# depth augmentation
_C.AUGMENTATION.DEPTH_BLUR = 'median_double' # gauss, box, median, median_double, none
_C.AUGMENTATION.DEPTH_BLUR_KERNEL_SIZE = 7

# pose augmentation
_C.AUGMENTATION.HAND_DROPOUT_MAX = 0 # 4
_C.AUGMENTATION.KNIFE_DROPOUT_MAX = 0 # 1

###############################
# OPTIMIZER ###################
###############################

# common #
_C.OPTIM = CN()
_C.OPTIM.GLR = 0.01
_C.OPTIM.DLR = 0.01
_C.OPTIM.PLR = 0.0001
_C.OPTIM.GNAME = "adam" # adam, sgd, radam
_C.OPTIM.DNAME = "sgd" # adam, sgd, radam
_C.OPTIM.PNAME = "radam" # adam, sgd, radam

# sgd
_C.OPTIM.SGD = CN()
_C.OPTIM.SGD.MOMENTUM = 0
_C.OPTIM.SGD.DAMPENING = 0
_C.OPTIM.SGD.WEIGHT_DECAY = 0
_C.OPTIM.SGD.NESTEROV = False 

# adam
_C.OPTIM.ADAM = CN()
_C.OPTIM.ADAM.BETA1 = 0.9
_C.OPTIM.ADAM.BETA2 = 0.999
_C.OPTIM.ADAM.EPS = 1e-8
_C.OPTIM.ADAM.WEIGHT_DECAY = 0

# radam 
_C.OPTIM.RADAM = CN()
_C.OPTIM.RADAM.BETA1 = 0.9
_C.OPTIM.RADAM.BETA2 = 0.999
_C.OPTIM.RADAM.EPS = 1e-8
_C.OPTIM.RADAM.WEIGHT_DECAY = 0

###############################
# SCHEDULER ###################
###############################

# common #
_C.SCHEDULER = CN()

# stepLR #
_C.SCHEDULER.STEPLR = CN()
_C.SCHEDULER.STEPLR.STEP_SIZE = 1000
_C.SCHEDULER.STEPLR.GAMMA = 0.1

###############################
# LOSS ########################
###############################

_C.G_LOSS_LIST = ['mse','l1-norm-reg']
_C.G_DEBUG_LOSS_LIST = ['gt_diff', 'subt_gt_diff']

_C.DISC = True # add adversarial loss tp generator and discriminator

_C.D_LOSS_LIST = ['w_gan', 'gp']

_C.PRED_LOSS_LIST = ['vae']

_C.LOSS = CN()

# MSE LOSS #
_C.LOSS.MSE = CN()
_C.LOSS.MSE.WEIGHT = 1.
_C.LOSS.MSE.DICT_ONLY = False

# L1 LOSS #
_C.LOSS.L1 = CN()
_C.LOSS.L1.WEIGHT = 1.
_C.LOSS.L1.DICT_ONLY = False

# L1-NORM for FRGAN #
_C.LOSS.L1_NORM_REG = CN()
_C.LOSS.L1_NORM_REG.WEIGHT = 0.01
_C.LOSS.L1_NORM_REG.DICT_ONLY = False

# GAN LOSS #
_C.LOSS.GAN = CN()
_C.LOSS.GAN.WEIGHT = 0.1
_C.LOSS.GAN.DICT_ONLY = False

# Gradient penalty #
_C.LOSS.GP = CN()
_C.LOSS.GP.WEIGHT = 0.01 # TODO change to 10

# VAE LOSS
_C.LOSS.CVAE = CN()
_C.LOSS.CVAE.WEIGHT = 1.
_C.LOSS.CVAE.RECONST_LOSS = 'mse' #L1
_C.LOSS.CVAE.KLD_WEIGHT = 1.

_C.LOSS.CVAE.IMAGE_LOSS = False
_C.LOSS.CVAE.IMAGE_LOSS_WEIGHT = 1.

_C.LOSS.CVAE.GAN_LOSS = False
_C.LOSS.CVAE.GAN_LOSS_WEIGHT = 1.

# rgb los
_C.LOSS.RGB = False
###############################
# MODEL #######################
###############################

_C.MODEL_NAME = 'FRGAN'
_C.MODEL = CN()
_C.PRED_NAME = 'cvae1'
_C.C_DISC = False # conditional discriminator
_C.LOAD_MODEL = 'all' # all, model_only
_C.USE_DEPTH = True

##### FRGAN #####
_C.MODEL.FRGAN = CN()
_C.MODEL.FRGAN_C = CN()

# FRGAN GEN #
_C.MODEL.FRGAN.GEN = CN()
_C.MODEL.FRGAN.GEN.H_DIM = 64
_C.MODEL.FRGAN.GEN.FILTER_SIZE = 3
_C.MODEL.FRGAN.GEN.MIN_FILTER_NUM = 64
_C.MODEL.FRGAN.GEN.MAX_FILETER_NUM = 256

_C.MODEL.FRGAN.GEN.LAST_ACTIVATION = 'none' # tanh relu
_C.MODEL.FRGAN.GEN.LAST_LAYER = 'heatmap' # normal, residual, heatmap
_C.MODEL.FRGAN.GEN.DEFORMABLE = False

# FRGAN DIS #
_C.MODEL.FRGAN.DIS = CN()
_C.MODEL.FRGAN.DIS.FILTER_SIZE = 3
_C.MODEL.FRGAN.DIS.MIN_FILTER_NUM = 64
_C.MODEL.FRGAN.DIS.MAX_FILETER_NUM = 256

# FRGAN_C GEN #
_C.MODEL.FRGAN_C.GEN = CN()
_C.MODEL.FRGAN_C.GEN.H_DIM = 64
_C.MODEL.FRGAN_C.GEN.FILTER_SIZE = 3
_C.MODEL.FRGAN_C.GEN.MIN_FILTER_NUM = 64
_C.MODEL.FRGAN_C.GEN.MAX_FILETER_NUM = 256

_C.MODEL.FRGAN_C.GEN.LAST_ACTIVATION = 'none' # tanh relu
_C.MODEL.FRGAN_C.GEN.LAST_LAYER = 'heatmap' # normal, residual, heatmap
_C.MODEL.FRGAN_C.GEN.DEFORMABLE = False
_C.MODEL.FRGAN_C.GEN.INPUT_MODE = 'future' # 'future'=(past, current, future), 'past'=(past,current), 'current'

##### VR_MP #####

_C.VR_MP = CN()

_C.VR_MP.SOFT_ARGMAX = False
_C.VR_MP.ACTIVATION = 'relu'
_C.VR_MP.NORM = 'none'

##### ASA_MP #####

_C.ASA_MP = CN()

_C.ASA_MP.ACTIVATION = 'relu'
_C.ASA_MP.NORM = 'none'

##### CVAE1 #####

_C.CVAE1 = CN()

_C.CVAE1.HIDDEN_DIM = 5
_C.CVAE1.ACTIVATION = 'lrelu'
_C.CVAE1.NORM = 'none'

_C.CVAE1.ENCODE_ALL_IMAGE = True
_C.CVAE1.ENCODE_ALL_POSE = True
_C.CVAE1.RESIDUAL_OUTPUT = False
_C.CVAE1.POSE_INPUT = True
_C.CVAE1.Z_DECODER = True
_C.CVAE1.WITHOUT_Z = False

_C.CVAE1.USE_RGB = True
_C.CVAE1.USE_DEPTH = True
_C.CVAE1.USE_POSE_IMAGE = True
_C.CVAE1.USE_ACTION = False

##### hourglass #####

_C.HOURGLASS = CN()

_C.HOURGLASS.NUM_BLOCK = 4
_C.HOURGLASS.NUM_DOWNSCALE = 4
_C.HOURGLASS.INTERMEDIATE_LOSS = True
_C.HOURGLASS.ACTIVATION = 'relu'
_C.HOURGLASS.NORM = 'none'
_C.HOURGLASS.BASE_FILTER = 256

# input
_C.HOURGLASS.INPUT_POSE = True
_C.HOURGLASS.INPUT_DEPTH = False
_C.HOURGLASS.INPUT_PAST = True # past rgb
_C.HOURGLASS.INPUT_Z = True
_C.HOURGLASS.INPUT_ROTATION = False
_C.HOURGLASS.INPUT_GRASP = False

# output
_C.HOURGLASS.PRED_RGB = False
_C.HOURGLASS.SINGLE_DEPTH = False
_C.HOURGLASS.PRED_ROTATION = False
_C.HOURGLASS.PRED_GRASP = False

##### VIDEO_HOURGLASS #####
_C.VIDEO_HOUR = CN()

_C.VIDEO_HOUR.TRAIN = False
_C.VIDEO_HOUR.MODE = 'pcf' #'pcf', 'pc', 'c'
_C.VIDEO_HOUR.DEPTH = False
_C.VIDEO_HOUR.LAST_LAYER = 'normal' # normal, residual, heatmap
_C.VIDEO_HOUR.MIN_FILTER_NUM = 64
_C.VIDEO_HOUR.MAX_FILTER_NUM = 256

##### sequence_hourglass #####
_C.SEQUENCE_HOUR =CN()

_C.SEQUENCE_HOUR.USE_VIDEOMODEL = True
_C.SEQUENCE_HOUR.MODE = 2