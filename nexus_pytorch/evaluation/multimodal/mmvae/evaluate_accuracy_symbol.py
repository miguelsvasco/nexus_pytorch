import os
import torch
import sacred
import random
import numpy as np
from tqdm import tqdm
import nexus_pytorch.evaluation.multimodal.mmvae.ingredients as ingredients
from nexus_pytorch.scenarios.multimodal_dataset.multimodal_dataset import MultimodalDataset
import nexus_pytorch.evaluation.multimodal.mmvae.model.training_utils as utils

ex = sacred.Experiment(
    'multimodal_mmvae_evaluate_accuracy_symbol',
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

def compute_accuracy(samples, target, classifier):

    if classifier is not None:
        log_ps = classifier(samples)
    else:
        log_ps = samples

    ps = torch.exp(log_ps)
    top_p, top_class = ps.topk(1, dim=1)
    equals = top_class == target.view(*top_class.shape)
    return torch.mean(equals.type(torch.FloatTensor))


def evaluate_recon(model, dataloader, n_samples, cuda):

    # Models in Eval mode
    model.eval()

    # Setup Average Meters
    # Single Modality CMI meters
    single_image_meter = utils.AverageMeter()
    single_trajectory_meter = utils.AverageMeter()
    single_sound_meter =  utils.AverageMeter()


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
            # From image
            _, _, _, cm_sym = model.generate(x_img=img_data)
            sym_acc = compute_accuracy(samples=cm_sym[1], target=labels, classifier=None)
            single_image_meter.update(sym_acc.item())

            # From sound
            _, _, _, cm_sym = model.generate(x_snd=snd_data)
            snd_acc = compute_accuracy(samples=cm_sym[1], target=labels, classifier=None)
            single_sound_meter.update(snd_acc.item())

            # From trajectory
            _, _, _, cm_sym = model.generate(x_trj=trj_data)
            trj_acc = compute_accuracy(samples=cm_sym[1], target=labels, classifier=None)
            single_trajectory_meter.update(trj_acc.item())



        # Compile Results
        sym_acc_scores_dic = {'single_image': single_image_meter.avg,
                              'single_sound': single_sound_meter.avg,
                              'single_trajectory': single_trajectory_meter.avg}

        single_sym_acc_results = [sym_acc_scores_dic['single_image'],
                                  sym_acc_scores_dic['single_sound'],
                                  sym_acc_scores_dic['single_trajectory']]

        print("\n Symbol Accuracy:")
        print("   * Single Modality = " + str(np.mean(single_sym_acc_results))
              + " +-" + str(np.std(single_sym_acc_results)))
        print("\n")

        return sym_acc_scores_dic




@ex.capture
def evaluate(model, _config, _run):

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
    sym_res = evaluate_recon(model, eval_dataset, n_samples, cuda)

    # Log values in mongodb
    _run.log_scalar('acc single image', sym_res['single_image'])
    _run.log_scalar('acc single sound', sym_res['single_sound'])
    _run.log_scalar('acc single trajectory', sym_res['single_trajectory'])

    # Save
    with open(os.path.join(exp_dir_path('evaluation'),  "sym_acc_" + str(drop_rate)
                                                       + "_" + str(seed) + ".pt"), 'wb') as f:
        torch.save(sym_res, f)

    ex.add_artifact(os.path.join(exp_dir_path('evaluation'),
                                  "sym_acc_" + str(drop_rate)
                                                       + "_" + str(seed) + ".pt"), name= "sym_acc_" + str(drop_rate)
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

    # Evaluate
    evaluate(model, _config)


if __name__ == '__main__':
    ex.run_commandline()