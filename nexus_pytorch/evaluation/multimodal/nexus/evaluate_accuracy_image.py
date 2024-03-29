import os
import torch
import sacred
import random
import numpy as np
from tqdm import tqdm
from nexus_pytorch.evaluation.classifiers.image.train import Image_Classifier
import nexus_pytorch.evaluation.multimodal.nexus.ingredients as ingredients
from nexus_pytorch.scenarios.multimodal_dataset.multimodal_dataset import MultimodalDataset
import nexus_pytorch.evaluation.multimodal.nexus.model.training_utils as utils

ex = sacred.Experiment(
    'multimodal_nexus_evaluate_accuracy_image',
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
    model, _ = utils.load_checkpoint(model_evaluation_config['file_local'], cuda)
    return model


def load_classifier(config, cuda):
    if cuda:
        checkpoint = torch.load('../../classifiers/image/trained_models/best_image_classifier_model.pth.tar')
    else:
        checkpoint = torch.load(
            '../../classifiers/image/trained_models/best_image_classifier_model.pth.tar',
            map_location=lambda storage, location: storage)

    model = Image_Classifier()
    model.load_state_dict(checkpoint['state_dict'])

    if cuda:
        model.cuda()

    return model


def compute_accuracy(samples, target, classifier):

    if classifier is not None:
        log_ps = classifier(samples)
    else:
        log_ps = samples

    ps = torch.exp(log_ps)
    top_p, top_class = ps.topk(1, dim=1)
    equals = top_class == target.view(*top_class.shape)
    return torch.mean(equals.type(torch.FloatTensor))


def evaluate_recon(model, img_class, dataloader, n_samples, cuda):

    # Models in Eval mode
    model.eval()
    img_class.eval()

    # Setup Average Meters
    # Single Modality CMI meters
    single_symbol_meter = utils.AverageMeter()
    single_trajectory_meter = utils.AverageMeter()
    single_sound_meter =  utils.AverageMeter()

    # Double Modality CMI meters
    double_symbol_sound_meter = utils.AverageMeter()
    double_symbol_trajectory_meter = utils.AverageMeter()
    double_sound_trajectory_meter = utils.AverageMeter()

    # Triple Modality CMI meters
    all_mods_meter = utils.AverageMeter()


    # Main Evaluation Loop
    with torch.no_grad():

        for batch_idx, data in enumerate(tqdm(dataloader)):

            # Original data
            img_data = data[1]
            trj_data = data[2]
            snd_data = data[3]
            sym_data = torch.nn.functional.one_hot(data[0], num_classes=10).float()
            labels = data[0]

            # To generate multiple samples
            labels = labels.repeat_interleave(repeats=n_samples, dim=0)
            img_data = img_data.repeat_interleave(repeats=n_samples, dim=0)
            trj_data = trj_data.repeat_interleave(repeats=n_samples, dim=0)
            snd_data = snd_data.repeat_interleave(repeats=n_samples, dim=0)
            sym_data = sym_data.repeat_interleave(repeats=n_samples, dim=0)

            if cuda:
                labels = labels.cuda()
                img_data = img_data.cuda()
                trj_data = trj_data.cuda()
                snd_data = snd_data.cuda()
                sym_data = sym_data.cuda()

            # Single Modality CMI
            # From symbol
            cm_img, _, _, _ = model.generate(x_sym=sym_data)
            sym_acc = compute_accuracy(samples=cm_img[1], target=labels, classifier=img_class)
            single_symbol_meter.update(sym_acc.item())

            # From sound
            cm_img, _, _, _ = model.generate(x_snd=snd_data)
            snd_acc = compute_accuracy(samples=cm_img[1], target=labels, classifier=img_class)
            single_sound_meter.update(snd_acc.item())

            # From trajectory
            cm_img, _, _, _ = model.generate(x_trj=trj_data)
            trj_acc = compute_accuracy(samples=cm_img[1], target=labels, classifier=img_class)
            single_trajectory_meter.update(trj_acc.item())

            # Double mod CMI encoding
            # From symbol and sound
            cm_img, _, _, _ = model.generate(x_snd=snd_data, x_sym=sym_data)
            sym_snd_acc = compute_accuracy(samples=cm_img[1], target=labels, classifier=img_class)
            double_symbol_sound_meter.update(sym_snd_acc.item())

            # From symbol and trajectory
            cm_img, _, _, _ = model.generate(x_trj=trj_data, x_sym=sym_data)
            sym_trj_acc = compute_accuracy(samples=cm_img[1], target=labels, classifier=img_class)
            double_symbol_trajectory_meter.update(sym_trj_acc.item())

            # From sound and trajectory
            cm_img, _, _, _ = model.generate(x_snd=snd_data, x_trj=trj_data)
            snd_trj_acc = compute_accuracy(samples=cm_img[1], target=labels, classifier=img_class)
            double_sound_trajectory_meter.update(snd_trj_acc.item())

            # Triple mod CMI encoding
            cm_img, _, _, _ = model.generate(x_sym=sym_data, x_snd=snd_data, x_trj=trj_data)
            all_mods_acc = compute_accuracy(samples=cm_img[1], target=labels, classifier=img_class)
            all_mods_meter.update(all_mods_acc.item())


        # Compile Results
        img_acc_scores_dic = {'single_symbol': single_symbol_meter.avg,
                              'single_sound': single_sound_meter.avg,
                              'single_trajectory': single_trajectory_meter.avg,
                              'double_symbol_sound': double_symbol_sound_meter.avg,
                              'double_symbol_trajectory': double_symbol_trajectory_meter.avg,
                              'double_sound_trajectory': double_sound_trajectory_meter.avg,
                              'all_mods': all_mods_meter.avg
                              }
        single_img_acc_results = [img_acc_scores_dic['single_symbol'],
                                  img_acc_scores_dic['single_sound'],
                                  img_acc_scores_dic['single_trajectory']]

        print("\n Image Accuracy:")
        print("   * Single Modality = " + str(np.mean(single_img_acc_results))
              + " +-" + str(np.std(single_img_acc_results)))
        print("   * All Modalities = " + str(img_acc_scores_dic['all_mods']))
        print("\n")

        return img_acc_scores_dic




@ex.capture
def evaluate(model, img_class, _config, _run):

    results_dir = log_dir_path('evaluation')
    training_config = _config['training']
    eval_config = _config['evaluation']
    cuda = _config['gpu']['cuda']
    drop_rate = training_config['nx_drop_rate']
    seed = training_config['seed']

    # Create Dataset
    dataset = MultimodalDataset(
        data_dir=exp_dir_path('../../../scenarios/multimodal_dataset/data'),
        batch_size=training_config['batch_size'],
        eval_samples=training_config['eval_samples'],
        validation_size=training_config['validation_size'],
        seed=seed)
    eval_dataset = dataset.get_test_loader(bsize=1)

    # Setup training
    n_samples = eval_config['eval_samples']
    _run.log_scalar('eval_samples', n_samples)

    # Evaluate image recognition
    img_res = evaluate_recon(model, img_class, eval_dataset, n_samples, cuda)

    # Log values in mongodb
    _run.log_scalar('acc single symbol', img_res['single_symbol'])
    _run.log_scalar('acc single sound', img_res['single_sound'])
    _run.log_scalar('acc single trajectory', img_res['single_trajectory'])
    _run.log_scalar('acc double symbol sound', img_res['double_symbol_sound'])
    _run.log_scalar('acc double symbol trajectory', img_res['double_symbol_trajectory'])
    _run.log_scalar('acc double sound trajectory', img_res['double_sound_trajectory'])
    _run.log_scalar('acc all mods', img_res['all_mods'])

    # Save
    with open(os.path.join(exp_dir_path('evaluation'),  "img_acc_" + str(drop_rate)
                                                       + "_" + str(seed) + ".pt"), 'wb') as f:
         torch.save(img_res, f)

    ex.add_artifact(os.path.join(exp_dir_path('evaluation'),
                                  "img_acc_" + str(drop_rate)
                                                       + "_" + str(seed) + ".pt"), name= "img_acc_" + str(drop_rate)
                                                       + "_" + str(seed) + ".pt")



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

    # Load classifiers
    print("Loading classifiers...")
    img_class = load_classifier(_config, _config['gpu']['cuda'])
    img_class.eval()

    # Evaluate
    evaluate(model, img_class, _config)


if __name__ == '__main__':
    ex.run_commandline()