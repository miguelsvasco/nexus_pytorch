import os
import shutil
import torch
import torchvision
import librosa
import torchaudio
from nexus_pytorch.evaluation.multimodal.mvae.model.model import MVAEModel
from nexus_pytorch.scenarios.multimodal_dataset.utils.trajectory_utils import *
from nexus_pytorch.evaluation.multimodal.SigmaVae.model.model import SigmaVAE


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

def get_specs(model_config):

    # Nexus dim
    nx_info = {
        'aggregate_function': model_config['nexus_aggregate_function'],
        'message_dim': model_config['nexus_message_dim'],
        'layer_sizes': model_config['nexus_layers'],
        'nexus_dim': model_config['nexus_dim']
    }

    # Image
    img_info = {
        'input_dim': model_config['image_channels'] * model_config['image_side'] * model_config['image_side'],
        'input_channels': model_config['image_channels'],
        'input_side': model_config['image_side'],
        'conv_layer_sizes': model_config['image_conv_layers'],
        'linear_layer_sizes': model_config['image_linear_layers'],
        'mod_latent_dim': model_config['image_mod_latent_dim'],
    }

    # Trajectory
    trj_info = {
        'input_dim': model_config['trajectory_size'],
        'linear_layer_sizes': model_config['trajectory_linear_layers'],
        'mod_latent_dim': model_config['trajectory_mod_latent_dim'],
    }

    # Sound
    snd_info = {
        'linear_layer_sizes': model_config['sound_linear_layers'],
        'mod_latent_dim': model_config['sound_mod_latent_dim'],
    }

    sym_info = {
        'input_dim': model_config['symbol_size'],
        'linear_layer_sizes': model_config['symbol_linear_layers'],
        'mod_latent_dim': model_config['symbol_mod_latent_dim'],
    }

    return nx_info, img_info, trj_info, snd_info, sym_info


def save_checkpoint(state,
                    is_best,
                    folder='./',
                    filename='mvae_checkpoint.pth.tar'):
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(
            os.path.join(folder, filename),
            os.path.join(folder, 'best_mvae_model.pth.tar'))


def load_sound_vae(cuda):

    if cuda:
            checkpoint = torch.load( '../SigmaVae/trained_models/best_sound_vae_model.pth.tar')
    else:
        checkpoint = torch.load(
            '../SigmaVae/trained_models/best_sound_vae_model.pth.tar',
            map_location=lambda storage, location: storage)

    snd_vae = SigmaVAE(latent_dim=128)
    snd_vae.load_state_dict(checkpoint['state_dict'])

    if cuda:
        snd_vae.cuda()

    return snd_vae

def load_checkpoint(checkpoint_file, use_cuda=False):
    if use_cuda:
        checkpoint = torch.load(checkpoint_file)
    else:
        checkpoint = torch.load(
            checkpoint_file, map_location=lambda storage, location: storage)

    model_config = checkpoint['model_config']
    nx_info, img_info, trj_info, snd_info, sym_info = get_specs(model_config)
    snd_vae = load_sound_vae(use_cuda)
    model = MVAEModel(nx_info=nx_info,
                        img_info=img_info,
                        trj_info=trj_info,
                        snd_info=snd_info,
                        sym_info=sym_info,
                        snd_vae=snd_vae,
                        use_cuda=use_cuda)
    model.load_state_dict(checkpoint['state_dict'])

    if use_cuda:
        model.cuda()

    return model, checkpoint['training_config']



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


class WarmUp(object):
    "Returns the value of the anneling factor to be used"

    def __init__(self, epochs=100, value=1.0):
        self.epoch = 0
        self.max_epoch = epochs
        self.value = value

    def get(self):

        if self.epoch >= self.max_epoch:
            return self.value
        else:
            return self.value*(float(self.epoch)/self.max_epoch)

    def update(self):
        self.epoch += 1



def save_image(data, path, filename, data_size=None):

    torchvision.utils.save_image(torchvision.utils.make_grid(data,
                                                             padding=5,
                                                             pad_value=.5,
                                                             nrow=data_size),
                                 os.path.join(path, filename))
    return

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

def save_sound(data, path, filename, sound_norm):

    for i in range(data.size(0)):
        # Unormalize data
        wave = data[i].cpu() * (sound_norm['max'] - sound_norm['min']) + sound_norm['min']

        # Permute channels and remove channel
        wave = wave.permute(0, 2, 1).squeeze(0)

        # DB to Power
        wave = librosa.db_to_power(wave)

        # Generate wave using Griffin-Lim algorithm
        sound_wav = librosa.feature.inverse.mel_to_audio(wave.squeeze(0).data.numpy(),
                                                         sr=16000,
                                                         n_iter=60)
        # Save data
        f_filename = filename + "_" + str(i) + ".wav"
        torchaudio.save(os.path.join(path, f_filename), torch.from_numpy(sound_wav) * np.iinfo(np.int16).max, 16000)

    return


def unstack_tensor(tensor, dim=0):

    tensor_lst = []
    for i in range(tensor.size(dim)):
        tensor_lst.append(tensor[i])
    tensor_unstack = torch.cat(tensor_lst, dim=0)
    return tensor_unstack


def gaussian_nll(mu, log_sigma, x):
    return 0.5 * torch.pow((x - mu) / log_sigma.exp(), 2) + log_sigma + 0.5 * np.log(2 * np.pi)