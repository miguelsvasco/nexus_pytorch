import os
import torch
import sacred
import torch.nn.functional as F
import nexus_pytorch.evaluation.multimodal.mvae.ingredients as ingredients
import nexus_pytorch.evaluation.multimodal.mvae.model.training_utils as utils
from torchvision.utils import save_image
from tqdm import tqdm

ex = sacred.Experiment(
    'multimodal_mvae_image_generation',
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
    cuda = _config['gpu']['cuda']

    # Setup Generation
    results_dir = exp_dir_path('sample_images')
    n_samples = model_eval_config['n_image_samples']

    # Sample cross-modality from symbol
    for i in tqdm(range(0, 10)):
        sample_v = generate_class(model, i, n_samples, cuda)
        # save image samples to filesystem
        save_image(sample_v.view(n_samples, 1, 28, 28),
                   os.path.join(results_dir, 'image_class_' + str(i) + '.png'))
        ex.add_artifact(
            os.path.join(results_dir, 'image_class_' + str(i) + '.png'),
            name='image_class_' + str(i) + '.png')


def generate_class(model, label_class, n_samples, cuda):

    symbol = F.one_hot(torch.tensor(label_class), 10).float().unsqueeze(0)
    symbol = symbol.repeat(n_samples, 1)

    if cuda:
        symbol = symbol.cuda()

    # Encode Mod Latents
    m_out, _, _, _= model.generate(x_sym=symbol)

    return m_out[1]


def get_model_by_config(config, cuda):

    model_evaluation_config = config['evaluation']
    model, _ = utils.load_checkpoint(model_evaluation_config['file_local'], cuda)
    return model

@ex.main
def main(_config, _run):
    os.makedirs(exp_dir_path('sample_images'), exist_ok=True)
    model = get_model_by_config(_config, _config['gpu']['cuda'])
    model.eval()
    sample(model, _config)


if __name__ == '__main__':
    ex.run_commandline()