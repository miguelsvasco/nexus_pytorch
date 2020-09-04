import os
import shutil
import torch
from nexus_pytorch.evaluation.preliminary.hierarchy.s_nexus.model.model import NexusModel


def get_specs(model_config):

    # Nexus dim
    nx_info = {
        'aggregate_function': model_config['nexus_aggregate_function'],
        'message_dim': model_config['nexus_message_dim'],
        'layer_sizes': model_config['nexus_layers'],
        'nexus_dim': model_config['nexus_dim']
    }

    # Setup dics
    img_info = {
        'input_dim': model_config['image_channels'] * model_config['image_side'] * model_config['image_side'],
        'input_channels': model_config['image_channels'],
        'input_side': model_config['image_side'],
        'conv_layer_sizes': model_config['image_conv_layers'],
        'linear_layer_sizes': model_config['image_linear_layers'],
        'mod_latent_dim': model_config['image_mod_latent_dim'],
    }

    sym_info = {
        'input_dim': model_config['symbol_size'],
        'linear_layer_sizes': model_config['symbol_linear_layers'],
        'mod_latent_dim': model_config['symbol_mod_latent_dim'],
    }

    return nx_info, img_info, sym_info


def save_checkpoint(state,
                    is_best,
                    folder='./',
                    filename='s_nexus_checkpoint.pth.tar'):
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(
            os.path.join(folder, filename),
            os.path.join(folder, 'best_s_nexus_model.pth.tar'))


def load_checkpoint(checkpoint_file, use_cuda=False):
    if use_cuda:
        checkpoint = torch.load(checkpoint_file)
    else:
        checkpoint = torch.load(
            checkpoint_file, map_location=lambda storage, location: storage)

    model_config = checkpoint['model_config']
    nx_info, v_info, sym_info = get_specs(model_config)
    model = NexusModel(nx_info=nx_info,
                        img_info=v_info,
                        sym_info=sym_info,
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