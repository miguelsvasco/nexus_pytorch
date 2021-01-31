import os
import torch
import sacred
import random
import numpy as np
from tqdm import tqdm
from nexus_pytorch.evaluation.classifiers.image.train import Image_Classifier
import nexus_pytorch.evaluation.preliminary.hierarchy.s_mvae.ingredients as ingredients
from nexus_pytorch.scenarios.multimodal_dataset.multimodal_dataset import MultimodalDataset
import nexus_pytorch.evaluation.preliminary.hierarchy.s_mvae.model.training_utils as utils

ex = sacred.Experiment(
    'hierarchy_s_mvae_evaluate_accuracy',
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
        checkpoint = torch.load('../../../classifiers/image/trained_models/best_image_classifier_model.pth.tar')
    else:
        checkpoint = torch.load(
            '../../../classifiers/image/trained_models/best_image_classifier_model.pth.tar',
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
    ## Image
    img_recon_meter = utils.AverageMeter()
    img_nx_recon_meter = utils.AverageMeter()
    cm_sym_img_recon_meter = utils.AverageMeter()

    ## Label
    sym_recon_meter = utils.AverageMeter()
    sym_nx_recon_meter = utils.AverageMeter()
    cm_img_sym_recon_meter = utils.AverageMeter()

    # Main Evaluation Loop
    with torch.no_grad():

        for batch_idx, modality_data in enumerate(tqdm(dataloader)):

            img_data = modality_data[1]
            img_data = img_data.repeat_interleave(repeats=n_samples, dim=0)
            sym_data_label = modality_data[0]
            sym_data_label = sym_data_label.repeat_interleave(repeats=n_samples, dim=0)
            sym_data = torch.nn.functional.one_hot(modality_data[0], num_classes=10).float()
            sym_data = sym_data.repeat_interleave(repeats=n_samples, dim=0)

            if cuda:
                img_data = img_data.cuda()
                sym_data = sym_data.cuda()
                sym_data_label = sym_data_label.cuda()

            # Modality Reconstruction
            img_out, sym_out = model.generate(x_v=img_data, x_s=sym_data)
            img_acc = compute_accuracy(samples=img_out, target=sym_data_label, classifier=img_class)
            sym_acc = compute_accuracy(samples=sym_out, target=sym_data_label, classifier=None)
            img_recon_meter.update(img_acc.item())
            sym_recon_meter.update(sym_acc.item())

            # Cross modality reconstruction
            ## Image to Symbol
            img_nx_img_out, img_nx_sym_out = model.generate(x_v=img_data)
            img_nx_acc = compute_accuracy(samples=img_nx_img_out, target=sym_data_label, classifier=img_class)
            img_sym_acc = compute_accuracy(samples=img_nx_sym_out, target=sym_data_label, classifier=None)
            cm_img_sym_recon_meter.update(img_sym_acc.item())
            img_nx_recon_meter.update(img_nx_acc.item())

            ## From Text
            sym_img_out, sym_nx_sym_out = model.generate(x_s=sym_data)
            sym_img_acc = compute_accuracy(samples=sym_img_out, target=sym_data_label, classifier=img_class)
            sym_nx_acc = compute_accuracy(samples=sym_nx_sym_out, target=sym_data_label, classifier=None)
            sym_nx_recon_meter.update(sym_nx_acc.item())
            cm_sym_img_recon_meter.update(sym_img_acc.item())

        # Compile Results
        img_res = {'recon': img_recon_meter.avg,  'nexus': img_nx_recon_meter.avg, 'cm': cm_sym_img_recon_meter.avg}
        sym_res = {'recon': sym_recon_meter.avg,  'nexus': sym_nx_recon_meter.avg, 'cm': cm_img_sym_recon_meter.avg}

        print("\nAccuracy = " + str((img_res['cm'] + sym_res['cm']) / 2.))
        print("\n")

        return img_res, sym_res




@ex.capture
def evaluate(model, img_class, _config, _run):

    results_dir = log_dir_path('evaluation')
    training_config = _config['training']
    eval_config = _config['evaluation']
    cuda = _config['gpu']['cuda']
    seed = training_config['seed']

    # Create Dataset
    dataset = MultimodalDataset(
        data_dir=exp_dir_path('../../../../scenarios/multimodal_dataset/data'),
        batch_size=training_config['batch_size'],
        eval_samples=training_config['eval_samples'],
        validation_size=training_config['validation_size'],
        seed=seed)

    eval_dataset = dataset.get_test_loader(bsize=1)

    # Setup training
    n_samples = eval_config['eval_samples']
    _run.log_scalar('eval_samples', n_samples)

    # Evaluate recognition
    img_res, sym_res = evaluate_recon(model, img_class, eval_dataset, n_samples, cuda)

    # Log values in mongodb
    _run.log_scalar('Image Recon', img_res['recon'])
    _run.log_scalar('Image Nexus Recon', img_res['nexus'])
    _run.log_scalar('Symbol to Image Recon', img_res['cm'])
    _run.log_scalar('Symbol Recon', sym_res['recon'])
    _run.log_scalar('Symbol Nexus Recon', sym_res['nexus'])
    _run.log_scalar('Image to Symbol Recon', sym_res['cm'])

    # Save
    with open(os.path.join(exp_dir_path('evaluation'), "img_recon_res.pt"), 'wb') as f:
        torch.save(img_res, f)
    ex.add_artifact(os.path.join(exp_dir_path('evaluation'),
                                 "img_recon_res.pt"), name='img_recon_res.pt')

    with open(os.path.join(exp_dir_path('evaluation'), "sym_recon_res.pt"), 'wb') as f:
        torch.save(sym_res, f)
    ex.add_artifact(os.path.join(exp_dir_path('evaluation'),
                                 "sym_recon_res.pt"), name='sym_recon_res.pt')
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

    # Load classifiers
    print("Loading classifiers...")
    img_class = load_classifier(_config, _config['gpu']['cuda'])
    img_class.eval()

    # Evaluate
    evaluate(model, img_class, _config)


if __name__ == '__main__':
    ex.run_commandline()