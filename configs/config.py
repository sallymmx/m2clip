# Path to the downloaded CLIP official weights.
# See: https://github.com/openai/CLIP/blob/a9b1bf5920416aaeaec965c25dd9e8f98c864f16/clip/clip.py#L30
CLIP_VIT_B16_PATH = '/slurm-files/wmm/models/ViT-B-16.pt'
CLIP_VIT_B32_PATH = '/slurm-files/wmm/models/ViT-B-32.pt'
CLIP_VIT_L14_PATH = '/slurm-files/wmm/models/ViT-L-14.pt'

# Whether cuDNN should be temporarily disable for 3D depthwise convolution.
# For some PyTorch builds the built-in 3D depthwise convolution may be much
# faster than the cuDNN implementation. You may experiment with your specific
# environment to find out the optimal option.
DWCONV3D_DISABLE_CUDNN = True

# Configuration of datasets. The required fields are listed for Something-something-v2 (ssv2)
# and Kinetics-400 (k400). Fill in the values to use the datasets, or add new datasets following
# these examples.
DATASETS = {
    'ssv2': dict(
        TRAIN_ROOT='',
        VAL_ROOT='',
        TRAIN_LIST='lists/sth/something_v2_rgb_train2.txt',
        VAL_LIST='lists/sth/something_v2_rgb_val2.txt',
        NUM_CLASSES=174,
    ),
    'k400': dict(
        TRAIN_ROOT='',
        VAL_ROOT='',
        TRAIN_LIST='lists/k400_train.txt',
        VAL_LIST='lists/k400_val.txt',
        NUM_CLASSES=400,
    ),
    'k400_128st': dict(
        TRAIN_ROOT='',
        VAL_ROOT='',
        TRAIN_LIST='lists/k4001/k400_tr_128v.txt',
        VAL_LIST='lists/k400_val.txt',
        NUM_CLASSES=400,
    ),
}
