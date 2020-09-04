import os
import torch
import sacred
import random
import numpy as np
import nexus_pytorch.evaluation.classifiers.trajectory.ingredients as ingredients
from nexus_pytorch.scenarios.multimodal_dataset.multimodal_dataset import MultimodalDataset
from nexus_pytorch.evaluation.classifiers.trajectory.train import Trajectory_Classifier

ex = sacred.Experiment(
    'trajectory_classifier_eval',
    ingredients=[
        ingredients.training_ingredient, ingredients.gpu_ingredient,
        ingredients.evaluation_ingredient
    ])


@ex.capture
def log_dir_path(folder, _config):
    return os.path.join('.', f'results/log_{_config["seed"]}/', folder)

@ex.capture
def exp_dir_path(folder, _config):
    return os.path.join('.', folder)


def load_classifier(cuda):
    if cuda:
        checkpoint = torch.load('./trained_models/best_trajectory_classifier_model.pth.tar')
    else:
        checkpoint = torch.load('./trained_models/best_trajectory_classifier_model.pth.tar',
            map_location=lambda storage, location: storage)

    model = Trajectory_Classifier()
    model.load_state_dict(checkpoint['state_dict'])

    if cuda:
        model.cuda()

    return model


def evaluate_acc(model, dataloader, cuda):
    model.eval()

    # Meters
    accuracy = 0
    with torch.no_grad():
        for batch_idx, modality_data in enumerate(dataloader):

            t_data = modality_data[1]
            labels = modality_data[0]

            if cuda:
                t_data = t_data.cuda()
                labels = labels.cuda()

            # Forward
            log_ps = model(t_data)
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))


    print("Test Accuracy: {:.3f}".format(100 * accuracy / len(dataloader)))

    return 100. * accuracy / len(dataloader)




@ex.capture
def evaluate(model, _config, _run):

    training_config = _config['training']
    cuda = _config['gpu']['cuda']

    # Create Dataset
    dataset = MultimodalDataset(
        data_dir=exp_dir_path('../../../scenarios/multimodal_dataset/data'),
        modalities=['trajectory'],
        batch_size=training_config['batch_size'],
        eval_samples=10,
        validation_size=0.1,
        seed=42)

    test_loader = dataset.test_loader

    accuracy = evaluate_acc(model, test_loader, cuda=cuda)
    _run.log_scalar('Test Accuracy', accuracy.item())

    return




@ex.main
def main(_config, _run):

    os.makedirs(log_dir_path('evaluation'), exist_ok=True)

    # Set seeds
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(_config['training']['seed'])
    np.random.seed(_config['training']['seed'])
    random.seed(_config['training']['seed'])
    torch.cuda.manual_seed(_config['training']['seed'])

    model = load_classifier(cuda = _config['gpu']['cuda'])
    model.eval()
    evaluate(model, _config)


if __name__ == '__main__':
    ex.run_commandline()