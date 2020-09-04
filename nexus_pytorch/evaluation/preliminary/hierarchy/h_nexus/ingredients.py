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
    epochs = 100
    learning_rate = 1e-3
    nx_drop_rate = 0.1

    # Capacity parameters
    lambda_image = 1.0
    lambda_symbol = 50.0
    beta_image = 1.0
    beta_symbol = 1.0
    gamma_image = 1.0
    gamma_symbol = 50.0
    beta_nexus = 1.0

    wup_mod_epochs = 20
    wup_nx_epochs = 20

    # Seed
    seed = 4


##########
# MODEL #
##########

model_ingredient = sacred.Ingredient('model')

@model_ingredient.config
def model_config():

    # Image Parameters
    image_channels = 1
    image_side = 28
    image_conv_layers = [32, 64]
    image_linear_layers = [128, 128]
    image_mod_latent_dim = 64

    # Symbol Parameters
    symbol_size = 10
    symbol_linear_layers = [128, 128, 128]
    symbol_mod_latent_dim = 5

    # Nexus Parameters
    nexus_dim = 16
    nexus_aggregate_function = 'mean_dropout'
    nexus_message_dim = 512
    nexus_layers = [512, 512]

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



##############
# Evaluation #
##############

evaluation_ingredient = sacred.Ingredient('evaluation')


@evaluation_ingredient.config
def evaluation_config():
    eval_samples = 1000
    file_local = 'trained_models/h_nexus_checkpoint.pth.tar'


##############
# Generation #
##############

generation_ingredient = sacred.Ingredient('generation')


@generation_ingredient.config
def generation_config():
    n_samples = 64