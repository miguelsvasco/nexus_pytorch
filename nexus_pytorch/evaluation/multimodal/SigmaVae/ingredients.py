import sacred

###############
#  Training   #
###############

training_ingredient = sacred.Ingredient('training')


@training_ingredient.config
def training_config():

    # Dataset parameters
    batch_size = 64
    validation_size = 0.1
    eval_samples = 15

    # Training Hyperparameters
    epochs = 2
    learning_rate = 1e-3

    # Capacity parameters
    # Sound
    lambda_sound = 1.0
    beta_sound = 10.0
    wup_mod_epochs = 0

    # Seed
    seed = 4


##########
# MODEL #
##########

model_ingredient = sacred.Ingredient('model')

@model_ingredient.config
def model_config():

    # Sound Parameters
    sound_linear_layers = []
    sound_mod_latent_dim = 128

model_debug_ingredient = sacred.Ingredient('model_debug')

@model_debug_ingredient.config
def debug_config():
    artifact_storage_interval = 1


########
# CUDA #
########

gpu_ingredient = sacred.Ingredient('gpu')


@gpu_ingredient.config
def gpu_config():
    cuda = True

