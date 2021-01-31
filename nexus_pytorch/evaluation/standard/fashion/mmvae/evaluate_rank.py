import os
import torch
import sacred
import random
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import nexus_pytorch.evaluation.standard.fashion.mmvae.ingredients as ingredients
from nexus_pytorch.scenarios.standard_dataset.standard_dataset import StandardDataset
import nexus_pytorch.evaluation.standard.fashion.mmvae.model.training_utils as t_utils
import nexus_pytorch.evaluation.standard.fashion.mmvae.model.evaluation_utils as e_utils
from nexus_pytorch.evaluation.autoencoders.fashion.train import Fashion_AE

ex = sacred.Experiment(
    'fashion_mmvae_evaluate_rank',
    ingredients=[
        ingredients.training_ingredient, ingredients.model_ingredient,
        ingredients.model_debug_ingredient, ingredients.gpu_ingredient,
        ingredients.evaluation_ingredient
    ])


@ex.capture
def log_dir_path(folder, _config):
    return os.path.join('.', f'results/log_{_config["seed"]}/', folder)

@ex.capture
def exp_dir_path(folder, _config):
    return os.path.join('.', folder)

def get_model_by_config(config, cuda):

    model_evaluation_config = config['evaluation']
    model, _ = t_utils.load_checkpoint(model_evaluation_config['file_local'], cuda)
    return model


def get_ae(cuda):

    # For now, we use only local files, and not mongodb (Error saving AE models in sacred...).
    if cuda:
        checkpoint = torch.load('../../../autoencoders/fashion/trained_models/best_fashion_ae_model.pth.tar')
    else:
        checkpoint = torch.load(
            '../../../autoencoders/fashion/trained_models/best_fashion_ae_model.pth.tar',
            map_location=lambda storage, location: storage)

    ae = Fashion_AE(b_dim=128)
    ae.load_state_dict(checkpoint['state_dict'])
    if cuda:
        ae = ae.cuda()
    return ae



@ex.capture
def evaluate(model, _config, _run):

    training_config = _config['training']
    cuda = _config['gpu']['cuda']
    drop_rate = training_config['nx_drop_rate']
    seed = training_config['seed']

    # Create Class Specific Dataset
    dataset = StandardDataset(
        dataset='fashion',
        data_dir='./data',
        batch_size=training_config['batch_size'],
        eval_samples=training_config['eval_samples'],
        validation_size=training_config['validation_size'],
        seed=training_config['seed'])
    eval_dataset = dataset.get_test_loader(bsize=1)

    # Load Vision Class-specific AE movel
    img_ae = get_ae(cuda)

    # Get real and fake images from dataset
    real_images = []
    fake_images = []
    with torch.no_grad():
        for idx, data in enumerate(tqdm(eval_dataset)):
            img = data[0]
            symbol = F.one_hot(torch.tensor(data[1]), 10).float()
            if cuda:
                img = img.cuda()
                symbol = symbol.cuda()
            real_images.append(img[0])
            # Encode Mod Latents
            img_out, _ = model.generate([None, symbol])
            nx_imgs = img_out[1]
            fake_images.append(nx_imgs[0])

    # Compute FID score
    fid_score = e_utils.compute_fid(real=real_images, fake=fake_images, ae_model=img_ae, batch_size=64)
    print("Rank SCORE = " + str(fid_score))

    # Log values in mongodb
    _run.log_scalar('Rank score', fid_score)

    # Save
    with open(os.path.join(exp_dir_path('evaluation'), "fid_single_score_" + str(drop_rate)
                                                       + "_" +str(seed) + ".txt"), 'w') as f:
        print("Rank Score= " + str(fid_score), file=f)

    with open(os.path.join(exp_dir_path('evaluation'), "fid_single_score_" + str(drop_rate)
                                                       + "_" +str(seed) + ".pt"), 'wb') as f:
        torch.save(fid_score, f)
    ex.add_artifact(os.path.join(exp_dir_path('evaluation'),
                                 "fid_single_score_" + str(drop_rate)
                                                       + "_" +str(seed) + ".pt"), name="fid_single_score_" + str(drop_rate)
                                                       + "_" +str(seed) + ".pt")

    return


@ex.main
def main(_config, _run):

    os.makedirs(exp_dir_path('evaluation'), exist_ok=True)

    # Set seeds
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(_config['training']['seed'])
    np.random.seed(_config['training']['seed'])
    random.seed(_config['training']['seed'])
    torch.cuda.manual_seed(_config['training']['seed'])

    # Load model
    print("Loading model...")
    model = get_model_by_config(_config, _config['gpu']['cuda'])
    model.eval()

    # Evaluate
    evaluate(model, _config)


if __name__ == '__main__':
    ex.run_commandline()