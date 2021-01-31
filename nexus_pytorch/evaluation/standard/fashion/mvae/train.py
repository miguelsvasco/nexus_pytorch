import os
import sys
import sacred
import random
import torch
import torchvision
import torch.nn.functional as F
import numpy as np
import nexus_pytorch.evaluation.standard.fashion.mvae.ingredients as ingredients
from nexus_pytorch.evaluation.standard.fashion.mvae.model.trainer import Trainer
from nexus_pytorch.evaluation.standard.fashion.mvae.model.model import MVAE
import nexus_pytorch.evaluation.standard.fashion.mvae.model.training_utils as t_utils
from nexus_pytorch.scenarios.standard_dataset.standard_dataset import StandardDataset

ex = sacred.Experiment(
    'fashion_mvae_train',
    ingredients=[
        ingredients.training_ingredient, ingredients.model_ingredient,
        ingredients.model_debug_ingredient, ingredients.gpu_ingredient
    ])

@ex.capture
def log_dir_path(folder, _config):
    return os.path.join('.', f'results/log_{_config["seed"]}', folder)


@ex.capture
def exp_dir_path(folder, _config):
    return os.path.join('.', folder)


class PostEpochCb(object):
    def __init__(self, model, dataset):
        self.model = model
        self.train_dataloader = dataset.train_loader
        self.val_dataloader = dataset.val_loader
        self.test_dataloader = dataset.get_test_loader(bsize=20)
        self.best_loss = sys.maxsize

    @ex.capture
    def _record_train_info(self, info, _run):
        _run.log_scalar('train_loss', info['loss'])
        _run.log_scalar('train_img_recon_loss', info['img_recon_loss'])
        _run.log_scalar('train_sym_recon_loss', info['sym_recon_loss'])
        _run.log_scalar('train_nx_prior_loss', info['nexus_prior_loss'])

    @ex.capture
    def _record_test_info(self, info, _run):
        _run.log_scalar('test_loss', info['loss'])
        _run.log_scalar('test_img_recon_loss', info['img_recon_loss'])
        _run.log_scalar('test_sym_recon_loss', info['sym_recon_loss'])
        _run.log_scalar('test_nx_prior_loss', info['nexus_prior_loss'])

    @ex.capture
    def _record_artifacts(self, info, _config):
        epoch = info['epoch']
        artifact_storage_interval = _config['model_debug'][
            'artifact_storage_interval']
        results_dir = log_dir_path('results')

        if epoch % artifact_storage_interval == 0:

            # Data
            with torch.no_grad():

                self.model.eval()
                test_data = next(iter(self.test_dataloader))
                img_data = test_data[0]
                sym_data = torch.nn.functional.one_hot(test_data[1], num_classes=10).float()

                if self.model.use_cuda:
                    img_data = img_data.cuda()
                    sym_data = sym_data.cuda()

                # Generate modalities
                img_out, sym_out  = self.model.generate(x_v=img_data,x_s=sym_data)

                img_nx_img_out, img_nx_sym_out = self.model.generate(x_v=img_data)

                sym_nx_img_out, sym_nx_sym_out = self.model.generate(x_s=sym_data)

                # Vision Recon
                img_comp = torch.cat([img_data.view(-1, 1, 28, 28).cpu(), img_out.view(-1, 1, 28, 28).cpu()])
                img_nx_img_comp = torch.cat([img_data.view(-1, 1, 28, 28).cpu(), img_nx_img_out.view(-1, 1, 28, 28).cpu()])
                sym_nx_img_comp = torch.cat([img_data.view(-1, 1, 28, 28).cpu(), sym_nx_img_out.view(-1, 1, 28, 28).cpu()])

                # Text Recon
                sym_res = np.argmax(F.log_softmax(sym_out, dim=-1).cpu().numpy(), axis=1).tolist()
                sym_res_str = ''
                for i, item in enumerate(sym_res):
                    sym_res_str += str(item) + " "

                img_nx_sym_out_res = np.argmax(F.log_softmax(img_nx_sym_out, dim=-1).cpu().numpy(), axis=1).tolist()
                img_nx_sym_out_res_str = ''
                for i, item in enumerate(img_nx_sym_out_res):
                    img_nx_sym_out_res_str += str(item) + " "

                sym_nx_sym_out_res = np.argmax(F.log_softmax(sym_nx_sym_out, dim=-1).cpu().numpy(), axis=1).tolist()
                sym_nx_sym_out_res_str = ''
                for i, item in enumerate(sym_nx_sym_out_res):
                    sym_nx_sym_out_res_str += str(item) + " "


            # Save data
            # MNIST
            torchvision.utils.save_image(torchvision.utils.make_grid(img_comp,
                                                                     padding=5,
                                                                     pad_value=.5,
                                                                     nrow=img_data.size(0)),
                                         os.path.join(results_dir, 'img_mod_e' + str(epoch) + '.png'))
            ex.add_artifact(os.path.join(results_dir, "img_mod_e" + str(epoch) + '.png'),
                            name="image_recon_e" + str(epoch) + '.png')


            torchvision.utils.save_image(torchvision.utils.make_grid(img_nx_img_comp,
                                                                     padding=5,
                                                                     pad_value=.5,
                                                                     nrow=img_data.size(0)),
                                         os.path.join(results_dir, 'img_nx_img_comp_e' + str(epoch) + '.png'))
            ex.add_artifact(os.path.join(results_dir, "img_nx_img_comp_e" + str(epoch) + '.png'),
                            name="image_nexus_image_comp_e" + str(epoch) + '.png')

            torchvision.utils.save_image(torchvision.utils.make_grid(sym_nx_img_comp,
                                                                     padding=5,
                                                                     pad_value=.5,
                                                                     nrow=img_data.size(0)),
                                         os.path.join(results_dir, 'sym_nx_img_comp_e' + str(epoch) + '.png'))
            ex.add_artifact(os.path.join(results_dir, "sym_nx_img_comp_e" + str(epoch) + '.png'),
                            name="symbol_nexus_image_comp_e" + str(epoch) + '.png')


            with open(os.path.join(results_dir,'sym_res_str_e' + str(epoch) + '.txt'), "w") as text_file:
                print(sym_res_str, file=text_file)
            ex.add_artifact(os.path.join(results_dir, "sym_res_str_e" + str(epoch) + '.txt'),
                            name= "symbol_recon_e" + str(epoch) + '.txt')

            with open(os.path.join(results_dir,'img_nx_sym_out_res_str_e' + str(epoch) + '.txt'), "w") as text_file:
                print(img_nx_sym_out_res_str, file=text_file)
            ex.add_artifact(os.path.join(results_dir, "img_nx_sym_out_res_str_e" + str(epoch) + '.txt'),
                            name= "image_nexus_symbol_recon_e" + str(epoch) + '.txt')

            with open(os.path.join(results_dir,'sym_nx_sym_out_res_str_e' + str(epoch) + '.txt'), "w") as text_file:
                print(sym_nx_sym_out_res_str, file=text_file)
            ex.add_artifact(os.path.join(results_dir, "sym_nx_sym_out_res_str_e" + str(epoch) + '.txt'),
                            name= "symbol_nexus_symbol_recon_e" + str(epoch) + '.txt')


    @ex.capture
    def _record_checkpoint(self, info, _config):
        test_info = info['test']
        loss = test_info['loss']
        is_best = loss < self.best_loss
        self.best_loss = min(loss, self.best_loss)

        # Find a way to make this modality agnostic..... -- TODO
        model_config = dict(_config['model'])
        training_config = dict(_config['training'])

        t_utils.save_checkpoint(
            {
                'state_dict': self.model.state_dict(),
                'best_loss': self.best_loss,
                'loss': test_info['loss'],
                'epoch': info['epoch'],
                'optimizer': info['optimizer'].state_dict(),
                'model_config': model_config,
                'training_config': training_config
            },
            is_best,
            folder=log_dir_path('trained_models'))

    @ex.capture
    def __call__(self, epoch_info, _config):
        self._record_train_info(epoch_info['train'])
        self._record_test_info(epoch_info['test'])
        self._record_artifacts(epoch_info)
        self._record_checkpoint(epoch_info)


def post_cb(info):
    ex.add_artifact(
        os.path.join(
            log_dir_path('trained_models'), 'mvae_checkpoint.pth.tar'),
        name='mvae_last_checkpoint.pth.tar')
    ex.add_artifact(
        os.path.join(
            log_dir_path('trained_models'), 'best_mvae_model.pth.tar'),
        name='best_mvae_model.pth.tar')


@ex.capture
def train(_config, _run):

    # Read configs
    model_config = _config['model']
    training_config = _config['training']
    gpu_config = _config['gpu']

    # Set seeds
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(training_config['seed'])
    np.random.seed(training_config['seed'])
    random.seed(training_config['seed'])
    torch.cuda.manual_seed(training_config['seed'])

    # Create Model
    nx_info, img_info, sym_info = t_utils.get_specs(model_config)
    model = MVAE(nx_info=nx_info,
                        img_info=img_info,
                        sym_info=sym_info,
                        use_cuda=gpu_config['cuda'])

    # Create trainer
    trainer = Trainer(model, training_config, gpu_config['cuda'])

    # Create Dataset
    dataset = StandardDataset(
        dataset='fashion',
        data_dir='./data',
        batch_size=training_config['batch_size'],
        eval_samples=training_config['eval_samples'],
        validation_size=training_config['validation_size'],
        seed=training_config['seed'])

    post_epoch_cb = PostEpochCb(model, dataset)

    trainer.train(epochs=training_config['epochs'], dataset=dataset, cuda=gpu_config['cuda'],
                  post_epoch_cb=post_epoch_cb, post_cb=post_cb)


@ex.main
def main(_config, _run):
    os.makedirs(log_dir_path('trained_models'), exist_ok=True)
    os.makedirs(log_dir_path('results'), exist_ok=True)

    train()


if __name__ == '__main__':
    ex.run_commandline()