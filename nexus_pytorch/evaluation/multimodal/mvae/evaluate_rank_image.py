import os
import torch
import sacred
import random
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import nexus_pytorch.evaluation.multimodal.mvae.ingredients as ingredients
from nexus_pytorch.scenarios.multimodal_dataset.multimodal_dataset import MultimodalDataset
import nexus_pytorch.evaluation.multimodal.mvae.model.training_utils as t_utils
import nexus_pytorch.evaluation.multimodal.mvae.model.evaluation_utils as e_utils
from nexus_pytorch.evaluation.autoencoders.image.train import Image_AE

ex = sacred.Experiment(
    'multimodal_mvae_evaluate_rank_image',
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
    if cuda:
        checkpoint = torch.load('../../autoencoders/image/trained_models/best_image_ae_model.pth.tar')
    else:
        checkpoint = torch.load(
            '../../autoencoders/image/trained_models/best_image_ae_model.pth.tar',
            map_location=lambda storage, location: storage)

    ae = Image_AE(b_dim=128)
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
    dataset = MultimodalDataset(
        data_dir=exp_dir_path('../../../scenarios/multimodal_dataset/data'),
        batch_size=training_config['batch_size'],
        eval_samples=training_config['eval_samples'],
        validation_size=training_config['validation_size'],
        seed=seed)
    eval_dataset = dataset.get_test_loader(bsize=1)

    # Load Vision Class-specific AE movel
    img_ae = get_ae(cuda)

    # Get real and fake images from dataset for every combination of modalities
    # Single modality CMI
    fake_images_from_symbol = []
    fake_images_from_sound = []
    fake_images_from_trajectory = []

    # Double modality CMI
    fake_images_from_symbol_sound = []
    fake_images_from_symbol_trajectory = []
    fake_images_from_sound_trajectory = []

    # Triple modality CMI
    fake_images_from_all_mods = []
    real_images = []
    with torch.no_grad():
        for idx, data in enumerate(tqdm(eval_dataset)):

            img_data = data[1]
            trj_data = data[2]
            snd_data = data[3]
            sym_data = torch.nn.functional.one_hot(data[0], num_classes=10).float()

            if cuda:
                img_data = img_data.cuda()
                trj_data = trj_data.cuda()
                snd_data = snd_data.cuda()
                sym_data = sym_data.cuda()

            real_images.append(img_data[0])

            # Single mod CMI encoding
            # From symbol
            cm_img, _, _, _ = model.generate(x_sym=sym_data)
            fake_images_from_symbol.append(cm_img[1][0])

            # From sound
            cm_img, _, _, _ = model.generate(x_snd=snd_data)
            fake_images_from_sound.append(cm_img[1][0])

            # From trajectory
            cm_img, _, _, _ = model.generate(x_trj=trj_data)
            fake_images_from_trajectory.append(cm_img[1][0])

            # Double mod CMI encoding
            # From symbol and sound
            cm_img, _, _, _ = model.generate(x_snd=snd_data, x_sym=sym_data)
            fake_images_from_symbol_sound.append(cm_img[1][0])

            # From symbol and trajectory
            cm_img, _, _, _ = model.generate(x_trj=trj_data, x_sym=sym_data)
            fake_images_from_symbol_trajectory.append(cm_img[1][0])

            # From sound and trajectory
            cm_img, _, _, _ = model.generate(x_snd=snd_data, x_trj=trj_data)
            fake_images_from_sound_trajectory.append(cm_img[1][0])

            # Triple mod CMI encoding
            cm_img, _, _, _ = model.generate(x_sym=sym_data, x_snd=snd_data, x_trj=trj_data)
            fake_images_from_all_mods.append(cm_img[1][0])


    # Compute FID scores
    img_fid_scores_dic = {}

    # SINGLE MOD CMI
    # From Symbol
    sym_fid_score = e_utils.compute_fid(real=real_images, fake=fake_images_from_symbol, ae_model=img_ae, batch_size=64)
    img_fid_scores_dic['single_symbol'] = sym_fid_score
    img_fid_scores_dic['single_mod'] = [sym_fid_score]

    # From Sound
    snd_fid_score = e_utils.compute_fid(real=real_images, fake=fake_images_from_sound, ae_model=img_ae, batch_size=64)
    img_fid_scores_dic['single_sound'] = snd_fid_score
    img_fid_scores_dic['single_mod'].append(snd_fid_score)

    # From Trajectory
    trj_fid_score = e_utils.compute_fid(real=real_images, fake=fake_images_from_trajectory, ae_model=img_ae, batch_size=64)
    img_fid_scores_dic['single_trajectory'] = trj_fid_score
    img_fid_scores_dic['single_mod'].append(trj_fid_score)

    # DOUBLE MOD CMI
    # From Symbol and Sound
    sym_snd_fid_score = e_utils.compute_fid(real=real_images, fake=fake_images_from_symbol_sound, ae_model=img_ae, batch_size=64)
    img_fid_scores_dic['double_symbol_sound'] = sym_snd_fid_score
    img_fid_scores_dic['double_mod'] = [sym_snd_fid_score]

    # From Symbol and Trajectory
    sym_trj_fid_score = e_utils.compute_fid(real=real_images, fake=fake_images_from_symbol_trajectory, ae_model=img_ae, batch_size=64)
    img_fid_scores_dic['double_symbol_trajectory'] = sym_trj_fid_score
    img_fid_scores_dic['double_mod'].append(sym_trj_fid_score)

    # From Trajectory
    snd_trj_fid_score = e_utils.compute_fid(real=real_images, fake=fake_images_from_sound_trajectory, ae_model=img_ae,
                                        batch_size=64)
    img_fid_scores_dic['double_sound_trajectory'] = snd_trj_fid_score
    img_fid_scores_dic['double_mod'].append(snd_trj_fid_score)

    # TRIPLE MOD CMI
    all_mod_fid_score = e_utils.compute_fid(real=real_images, fake=fake_images_from_all_mods, ae_model=img_ae,
                                            batch_size=64)
    img_fid_scores_dic['all_mods'] = all_mod_fid_score

    single_img_dis_results = [img_fid_scores_dic['single_symbol'],
                              img_fid_scores_dic['single_sound'],
                              img_fid_scores_dic['single_trajectory']]

    print("\n Image Rank:")
    print("   * Single Modality = " + str(np.mean(single_img_dis_results))
          + " +-" + str(np.std(single_img_dis_results)))
    print("   * All Modalities = " + str(img_fid_scores_dic['all_mods']))
    print("\n")

    # Log values in mongodb
    _run.log_scalar('fid single symbol', sym_fid_score)
    _run.log_scalar('fid single sound', snd_fid_score)
    _run.log_scalar('fid single trajectory', trj_fid_score)
    _run.log_scalar('fid double symbol sound', sym_snd_fid_score)
    _run.log_scalar('fid double symbol trajectory', sym_trj_fid_score)
    _run.log_scalar('fid double sound trajectory', snd_trj_fid_score)
    _run.log_scalar('fid all mods', all_mod_fid_score)

    # Save
    with open(os.path.join(exp_dir_path('evaluation'), "img_fid_" + str(drop_rate)
                                                       + "_" + str(seed) + ".txt"), 'w') as f:
        print(img_fid_scores_dic, file=f)

    with open(os.path.join(exp_dir_path('evaluation'), "img_fid_" + str(drop_rate)
                                                       + "_" + str(seed) + ".pt"), 'wb') as f:
        torch.save(img_fid_scores_dic, f)

    ex.add_artifact(os.path.join(exp_dir_path('evaluation'),
                                 "img_fid_" + str(drop_rate)
                                 + "_" + str(seed) + ".pt"), name="img_fid_" + str(drop_rate)
                                                                  + "_" + str(seed) + ".pt")
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