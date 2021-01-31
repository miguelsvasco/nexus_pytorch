import sacred

###############
#  Training   #
###############

training_ingredient = sacred.Ingredient('training')


@training_ingredient.config
def training_config():

    experiment = 'fashion_classifier'

    # Dataset parameters
    batch_size = 64

    # Training Hyperparameters
    epochs = 100
    learning_rate = 1e-3

    # Seed
    seed = 4



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

    file = 'best_mnist_classifier_model.pth.tar'
