import os
import torch
import sacred
import random
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import nexus_pytorch.evaluation.multimodal.nexus.ingredients as ingredients
from nexus_pytorch.scenarios.multimodal_dataset.multimodal_dataset import MultimodalDataset
import nexus_pytorch.evaluation.multimodal.nexus.model.training_utils as t_utils
import nexus_pytorch.evaluation.multimodal.nexus.model.evaluation_utils as e_utils
from nexus_pytorch.evaluation.autoencoders.sound.train import Sound_AE

ex = sacred.Experiment(
    'multimodal_nexus_evaluate_dissimilarity_sound',
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
        checkpoint = torch.load('../../autoencoders/sound/trained_models/best_sound_ae_model.pth.tar')
    else:
        checkpoint = torch.load(
            '../../autoencoders/sound/trained_models/best_sound_ae_model.pth.tar',
            map_location=lambda storage, location: storage)

    ae = Sound_AE(b_dim=512)
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
    snd_ae = get_ae(cuda)

    # Get real and fake sounds from dataset for every combination of modalities
    # Single modality CMI
    fake_sounds_from_symbol = []
    fake_sounds_from_image = []
    fake_sounds_from_trajectory = []

    # Double modality CMI
    fake_sounds_from_symbol_image = []
    fake_sounds_from_symbol_trajectory = []
    fake_sounds_from_image_trajectory = []

    # Triple modality CMI
    fake_sounds_from_all_mods = []
    real_sounds = []
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

            real_sounds.append(snd_data[0])

            # Single mod CMI encoding
            # From symbol
            _, _, cm_snd, _ = model.generate(x_sym=sym_data)
            fake_sounds_from_symbol.append(cm_snd[1][0])

            # From image
            _, _, cm_snd, _ = model.generate(x_img=img_data)
            fake_sounds_from_image.append(cm_snd[1][0])

            # From trajectory
            _, _, cm_snd, _ = model.generate(x_trj=trj_data)
            fake_sounds_from_trajectory.append(cm_snd[1][0])

            # Double mod CMI encoding
            # From symbol and image
            _, _, cm_snd, _ = model.generate(x_img=img_data, x_sym=sym_data)
            fake_sounds_from_symbol_image.append(cm_snd[1][0])

            # From symbol and trajectory
            _, _, cm_snd, _ = model.generate(x_trj=trj_data, x_sym=sym_data)
            fake_sounds_from_symbol_trajectory.append(cm_snd[1][0])

            # From image and trajectory
            _, _, cm_snd, _ = model.generate(x_img=img_data, x_trj=trj_data)
            fake_sounds_from_image_trajectory.append(cm_snd[1][0])

            # Triple mod CMI encoding
            _, _, cm_snd, _ = model.generate(x_sym=sym_data, x_img=img_data, x_trj=trj_data)
            fake_sounds_from_all_mods.append(cm_snd[1][0])


    # Compute FID scores
    snd_fid_scores_dic = {}

    # SINGLE MOD CMI
    # From Symbol
    sym_fid_score = e_utils.compute_fid(real=real_sounds, fake=fake_sounds_from_symbol, ae_model=snd_ae, batch_size=64)
    snd_fid_scores_dic['single_symbol'] = sym_fid_score
    snd_fid_scores_dic['single_mod'] = [sym_fid_score]

    # From Image
    img_fid_score = e_utils.compute_fid(real=real_sounds, fake=fake_sounds_from_image, ae_model=snd_ae, batch_size=64)
    snd_fid_scores_dic['single_image'] = img_fid_score
    snd_fid_scores_dic['single_mod'].append(img_fid_score)

    # From Trajectory
    trj_fid_score = e_utils.compute_fid(real=real_sounds, fake=fake_sounds_from_trajectory, ae_model=snd_ae, batch_size=64)
    snd_fid_scores_dic['single_trajectory'] = trj_fid_score
    snd_fid_scores_dic['single_mod'].append(trj_fid_score)

    # DOUBLE MOD CMI
    # From Symbol and Image
    sym_img_fid_score = e_utils.compute_fid(real=real_sounds, fake=fake_sounds_from_symbol_image, ae_model=snd_ae, batch_size=64)
    snd_fid_scores_dic['double_symbol_image'] = sym_img_fid_score
    snd_fid_scores_dic['double_mod'] = [sym_img_fid_score]

    # From Symbol and Trajectory
    sym_trj_fid_score = e_utils.compute_fid(real=real_sounds, fake=fake_sounds_from_symbol_trajectory, ae_model=snd_ae, batch_size=64)
    snd_fid_scores_dic['double_symbol_trajectory'] = sym_trj_fid_score
    snd_fid_scores_dic['double_mod'].append(sym_trj_fid_score)

    # From Image and Trajectory
    img_trj_fid_score = e_utils.compute_fid(real=real_sounds, fake=fake_sounds_from_image_trajectory, ae_model=snd_ae,
                                        batch_size=64)
    snd_fid_scores_dic['double_image_trajectory'] = img_trj_fid_score
    snd_fid_scores_dic['double_mod'].append(img_trj_fid_score)

    # TRIPLE MOD CMI
    all_mod_fid_score = e_utils.compute_fid(real=real_sounds, fake=fake_sounds_from_all_mods, ae_model=snd_ae,
                                            batch_size=64)
    snd_fid_scores_dic['all_mods'] = all_mod_fid_score



    single_snd_dis_results = [snd_fid_scores_dic['single_symbol'],
                              snd_fid_scores_dic['single_image'],
                              snd_fid_scores_dic['single_trajectory']]

    print("\n Sound Dissimilarity:")
    print("   * Single Modality = " + str(np.mean(single_snd_dis_results))
          + " +-" + str(np.std(single_snd_dis_results)))
    print("   * All Modalities = " + str(snd_fid_scores_dic['all_mods']))
    print("\n")

    # Log values in mongodb
    _run.log_scalar('fid single symbol', sym_fid_score)
    _run.log_scalar('fid single image', img_fid_score)
    _run.log_scalar('fid single trajectory', trj_fid_score)
    _run.log_scalar('fid double symbol image', sym_img_fid_score)
    _run.log_scalar('fid double symbol trajectory', sym_trj_fid_score)
    _run.log_scalar('fid double image trajectory', img_trj_fid_score)
    _run.log_scalar('fid all mods', all_mod_fid_score)

    # Save
    with open(os.path.join(exp_dir_path('evaluation'), "snd_fid_" + str(drop_rate)
                                                       + "_" +str(seed) + ".txt"), 'w') as f:
        print(snd_fid_scores_dic, file=f)

    with open(os.path.join(exp_dir_path('evaluation'), "snd_fid_" + str(drop_rate)
                                                       + "_" +str(seed) + ".pt"), 'wb') as f:
        torch.save(snd_fid_scores_dic, f)

    ex.add_artifact(os.path.join(exp_dir_path('evaluation'),
                                 "snd_fid_" + str(drop_rate)
                                                       + "_" +str(seed) + ".pt"), name="snd_fid_" + str(drop_rate)
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