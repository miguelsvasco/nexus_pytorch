import os
import torch
import sacred
from torch.autograd import Variable
import torch.nn.functional as F
import nexus_pytorch.evaluation.multimodal.nexus.ingredients as ingredients
import nexus_pytorch.evaluation.multimodal.nexus.model.training_utils as utils
from nexus_pytorch.scenarios.multimodal_dataset.multimodal_dataset import MultimodalDataset
from tqdm import tqdm

ex = sacred.Experiment(
    'multimodal_nexus_sound_generation',
    ingredients=[
        ingredients.training_ingredient, ingredients.model_ingredient,
        ingredients.model_debug_ingredient, ingredients.gpu_ingredient,
        ingredients.evaluation_ingredient,
        ingredients.generation_ingredient
    ])


@ex.capture
def log_dir_path(folder, _config):
    return os.path.join('.', f'results/log_{_config["seed"]}/', folder)

@ex.capture
def exp_dir_path(folder, _config):
    return os.path.join('.', folder)



@ex.capture
def sample(model, _config, _run):
    model_eval_config = _config['generation']
    training_config = _config['training']
    cuda = _config['gpu']['cuda']

    # Setup Generation
    results_dir = exp_dir_path('sample_sounds')
    n_samples = model_eval_config['n_sound_samples']

    # Create Dataset
    dataset = MultimodalDataset(
        data_dir=exp_dir_path('../../../scenarios/multimodal_dataset/data'),
        modalities=['sound'],
        batch_size=training_config['batch_size'],
        eval_samples=training_config['eval_samples'],
        validation_size=training_config['validation_size'],
        seed=training_config['seed'])

    sound_norm = dataset.get_sound_normalization()


    # Sample cross-modality from symbol
    for i in tqdm(range(0, 10)):
        samples_sound = generate_class(model, i, n_samples, cuda)
        utils.save_sound(data=samples_sound, path=results_dir, filename="sound_class_"+ str(i),sound_norm=sound_norm)


def generate_class(model, label_class, n_samples, cuda):

    symbol = F.one_hot(torch.tensor(label_class), 10).float().unsqueeze(0)
    symbol = symbol.repeat(n_samples, 1)

    if cuda:
        symbol = symbol.cuda()

    # Encode Mod Latents
    _, _, s_out, _= model.generate(x_sym=symbol)

    return s_out[1]




def get_model_by_config(config, cuda):

    model_evaluation_config = config['evaluation']
    model, _ = utils.load_checkpoint(model_evaluation_config['file_local'], cuda)
    return model

@ex.main
def main(_config, _run):
    os.makedirs(exp_dir_path('sample_sounds'), exist_ok=True)
    model = get_model_by_config(_config, _config['gpu']['cuda'])
    model.eval()
    sample(model, _config)


if __name__ == '__main__':
    ex.run_commandline()