from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import sacred
import random
import numpy as np
import shutil
import sys
import nexus_pytorch.evaluation.autoencoders.trajectory.ingredients as ingredients
import torchvision
from nexus_pytorch.scenarios.multimodal_dataset.multimodal_dataset import MultimodalDataset
from nexus_pytorch.scenarios.multimodal_dataset.utils.trajectory_utils import *

ex = sacred.Experiment(
    'trajectory_autoencoder',
    ingredients=[ingredients.gpu_ingredient, ingredients.training_ingredient,
                 ingredients.model_debug_ingredient
                 ])


@ex.capture
def log_dir_path(folder, _config):
    return os.path.join('.', f'results/log_{_config["seed"]}', folder)


@ex.capture
def exp_dir_path(folder, _config):
    return os.path.join('.', folder)


def generate_image_from_trajectory(traj, tmp_path):

    # Plot Trajectory in color and save image
    fig, ax = plot_single_stroke_digit_evaluation(traj)
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(os.path.join(tmp_path,'tmp.png'), bbox_inches=extent, dpi=100)
    plt.close(fig)

    # Get image in greyscale
    g_img = get_greyscale_image(os.path.join(tmp_path,'tmp.png'))
    np_img = np.asarray(g_img)
    return torch.from_numpy(np_img).unsqueeze(0).float()/float(255.), g_img  # Normalize data!



def save_trajectory(data, path, filename, traj_norm, tmp_path="./"):
    trajs = []
    for i in range(data.size(0)):

        # Unnormalize data
        trajectory = data[i] * (traj_norm['max'] - traj_norm['min']) + traj_norm['min']

        # Generate image of trajectory
        trajs.append(generate_image_from_trajectory(traj=trajectory.cpu(), tmp_path=tmp_path)[0])

    t_trajs = torch.stack(trajs, dim=0)
    torchvision.utils.save_image(torchvision.utils.make_grid(t_trajs,
                                                             padding=5,
                                                             pad_value=.5,
                                                             nrow=t_trajs.size(0)),
                                 os.path.join(path, filename))
    return


def save_checkpoint(state,
                    is_best,
                    folder='./'):


    filename = 'trajectory_ae_checkpoint.pth.tar'
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(
            os.path.join(folder, filename),
            os.path.join(folder, 'best_trajectory_ae_model.pth.tar'))


@ex.capture
def record_checkpoint(model, loss, best_loss, optimizer, epoch, is_best):

    save_checkpoint(
        {
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'loss': loss,
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
        },
        is_best,
        folder=log_dir_path('trained_models'))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# Extra Components
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Trajectory_AE(nn.Module):
    def __init__(self, b_dim):
        super(Trajectory_AE, self).__init__()
        self.b_dim = b_dim
        self.encoder = nn.Sequential(nn.Linear(200, 512),
                                  nn.BatchNorm1d(512),
                                  nn.LeakyReLU(),
                                  nn.Linear(512, 512),
                                  nn.BatchNorm1d(512),
                                  nn.LeakyReLU(),
                                  nn.Linear(512, 512),
                                  nn.BatchNorm1d(512),
                                  nn.LeakyReLU(),
                                  nn.Linear(512, b_dim))

        self.decoder = nn.Sequential(nn.Linear(b_dim, 512),
                                  nn.BatchNorm1d(512),
                                  nn.LeakyReLU(),
                                  nn.Linear(512, 512),
                                  nn.BatchNorm1d(512),
                                  nn.LeakyReLU(),
                                  nn.Linear(512, 512),
                                  nn.BatchNorm1d(512),
                                  nn.LeakyReLU(),
                                  nn.Linear(512, 200),
                                     nn.Sigmoid())


    def encode(self, x):
            return self.encoder(x)

    def decode(self, h):
            return self.decoder(h)

    def forward(self, x):
        b = self.encode(x)
        return self.decode(b)


def train_epoch(model, train_loader, optimizer, epoch, cuda):
    model.train()

    # Meters
    loss_meter = AverageMeter()

    for batch_idx, data in enumerate(train_loader):

        trajectory = data[1]
        bs = trajectory.size(0)

        if cuda:
            trajectory = trajectory.cuda()

        optimizer.zero_grad()
        output = model(trajectory)
        loss = torch.mean(torch.sum(F.mse_loss(output.view(output.size(0), -1),
                                               trajectory.view(trajectory.size(0), -1),
                             reduction='none'), dim=-1))
        loss.backward()
        optimizer.step()

        # Update meters
        loss_meter.update(loss.item(), bs)

        # log every 100 batches
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Loss: {loss_meter.avg:.6f}')

    print(f'====> Epoch: {epoch}\t' f'Loss: {loss_meter.avg:.4f}')

    return loss_meter.avg


def test_epoch(model, test_loader, cuda):
    model.eval()

    # Meters
    loss_meter = AverageMeter()

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):

            trajectory = data[1]
            bs = trajectory.size(0)
            if cuda:
                trajectory = trajectory.cuda()

            # Forward
            output = model(trajectory)
            loss = torch.mean(torch.sum(F.mse_loss(output.view(output.size(0), -1),
                                                   trajectory.view(trajectory.size(0), -1),
                                                   reduction='none'), dim=-1))

            # Update meters
            loss_meter.update(loss.item(), bs)

    print(f'====>Test Loss: {loss_meter.avg:.4f}')

    return loss_meter.avg


@ex.capture
def train(_config, _run):

    # Read configs
    training_config = _config['training']
    device = torch.device("cuda" if _config['gpu']['cuda'] else "cpu")
    results_dir = log_dir_path('results')
    artifact_storage_interval = _config['model_debug'][
        'artifact_storage_interval']

    # Set seeds
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(training_config['seed'])
    np.random.seed(training_config['seed'])
    random.seed(training_config['seed'])
    torch.cuda.manual_seed(training_config['seed'])

    # Create Classifier
    model = Trajectory_AE(b_dim=training_config['b_dim']).to(device)
    epochs = training_config['epochs']

    # Create Dataset
    dataset = MultimodalDataset(
        modalities=['trajectory'],
        data_dir=exp_dir_path('../../../scenarios/multimodal_dataset/data'),
        batch_size=training_config['batch_size'],
        eval_samples=10,
        validation_size=0.1,
        seed=training_config['seed'])

    trj_norm = dataset.get_traj_normalization()

    train_loader, val_loader = dataset.train_loader, dataset.val_loader
    test_loader = dataset.get_test_loader(bsize=20)

    # Training objects
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config['learning_rate'])
    best_loss = sys.maxsize

    for epoch in range(1, epochs + 1):

        train_loss = train_epoch(model, train_loader, optimizer, epoch, cuda=_config['gpu']['cuda'])
        val_loss = test_epoch(model, val_loader, cuda=_config['gpu']['cuda'])

        _run.log_scalar('train_loss', train_loss)
        _run.log_scalar('val_loss', val_loss)

        # Best Loss
        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)
        record_checkpoint(model=model, loss=val_loss, best_loss=best_loss,
                          optimizer=optimizer, epoch=epoch, is_best=is_best)


        if epoch % artifact_storage_interval == 0:

            # Data
            with torch.no_grad():

                model.eval()
                data = next(iter(test_loader))[1].to(device)

                # Generate modalities
                out = model(data)

                save_trajectory(data=data, path=results_dir, filename='trj_og_e' + str(epoch) + '.png',
                                        traj_norm=trj_norm,
                                        tmp_path=results_dir)
                ex.add_artifact(os.path.join(results_dir, "trj_og_e" + str(epoch) + '.png'),
                                name="trajectory_og_e" + str(epoch) + '.png')

                save_trajectory(data=out, path=results_dir, filename='trj_mod_e' + str(epoch) + '.png',
                                        traj_norm=trj_norm,
                                        tmp_path=results_dir)
                ex.add_artifact(os.path.join(results_dir, "trj_mod_e" + str(epoch) + '.png'),
                                name="trajectory_recon_e" + str(epoch) + '.png')



    # Final Saving
    ex.add_artifact(
        os.path.join(
            log_dir_path('trained_models'), 'trajectory_ae_checkpoint.pth.tar'),
        name='trajectory_ae_last_checkpoint.pth.tar')
    ex.add_artifact(
        os.path.join(
            log_dir_path('trained_models'), 'best_trajectory_ae_model.pth.tar'),
        name='best_trajectory_ae_model.pth.tar')




@ex.main
def main(_config, _run):
    os.makedirs(log_dir_path('trained_models'), exist_ok=True)
    os.makedirs(log_dir_path('results'), exist_ok=True)
    train()


if __name__ == '__main__':
    ex.run_commandline()