import sacred

###############
#  Training   #
###############

training_ingredient = sacred.Ingredient('training')


@training_ingredient.config
def training_config():

    # Dataset parameters
    batch_size = 16

    # Training Hyperparameters
    epochs = 100
    learning_rate = 1e-3
    b_dim = 64
    image_class = 0

    # Seed
    seed = 4


model_debug_ingredient = sacred.Ingredient('model_debug')

@model_debug_ingredient.config
def debug_config():
    artifact_storage_interval = 10



########
# CUDA #
########

gpu_ingredient = sacred.Ingredient('gpu')


@gpu_ingredient.config
def gpu_config():
    cuda = True