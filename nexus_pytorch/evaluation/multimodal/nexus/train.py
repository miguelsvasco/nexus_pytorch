import os
import sys
import sacred
import random
import torch
import torchvision
import torch.nn.functional as F
import numpy as np
import nexus_pytorch.evaluation.multimodal.nexus.ingredients as ingredients
from nexus_pytorch.evaluation.multimodal.nexus.model.trainer import Trainer
from nexus_pytorch.evaluation.multimodal.nexus.model.model import NexusModel
import nexus_pytorch.evaluation.multimodal.nexus.model.training_utils as t_utils
from nexus_pytorch.scenarios.multimodal_dataset.multimodal_dataset import MultimodalDataset

ex = sacred.Experiment(
    'multimodal_nexus_train',
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
        self.snd_norm = dataset.get_sound_normalization()
        self.trj_norm = dataset.get_traj_normalization()
        self.train_dataloader = dataset.train_loader
        self.val_dataloader = dataset.val_loader
        self.test_dataloader = dataset.get_test_loader(bsize=10)
        self.best_loss = sys.maxsize

    @ex.capture
    def _record_train_info(self, info, _run):
        _run.log_scalar('train_loss', info['loss'])
        _run.log_scalar('train_img_recon_loss', info['img_recon_loss'])
        _run.log_scalar('train_trj_recon_loss', info['trj_recon_loss'])
        _run.log_scalar('train_sym_recon_loss', info['sym_recon_loss'])
        _run.log_scalar('train_img_mod_prior_loss', info['img_mod_prior_loss'])
        _run.log_scalar('train_trj_mod_prior_loss', info['trj_mod_prior_loss'])
        _run.log_scalar('train_sym_mod_prior_loss', info['sym_mod_prior_loss'])
        _run.log_scalar('train_img_nx_recon_loss', info['img_nx_recon_loss'])
        _run.log_scalar('train_trj_nx_recon_loss', info['trj_nx_recon_loss'])
        _run.log_scalar('train_snd_nx_recon_loss', info['snd_nx_recon_loss'])
        _run.log_scalar('train_sym_nx_recon_loss', info['sym_nx_recon_loss'])
        _run.log_scalar('train_nx_prior_loss', info['nexus_prior_loss'])

    @ex.capture
    def _record_test_info(self, info, _run):
        _run.log_scalar('test_loss', info['loss'])
        _run.log_scalar('test_img_recon_loss', info['img_recon_loss'])
        _run.log_scalar('test_trj_recon_loss', info['trj_recon_loss'])
        _run.log_scalar('test_sym_recon_loss', info['sym_recon_loss'])
        _run.log_scalar('test_img_mod_prior_loss', info['img_mod_prior_loss'])
        _run.log_scalar('test_trj_mod_prior_loss', info['trj_mod_prior_loss'])
        _run.log_scalar('test_sym_mod_prior_loss', info['sym_mod_prior_loss'])
        _run.log_scalar('test_img_nx_recon_loss', info['img_nx_recon_loss'])
        _run.log_scalar('test_trj_nx_recon_loss', info['trj_nx_recon_loss'])
        _run.log_scalar('test_snd_nx_recon_loss', info['snd_nx_recon_loss'])
        _run.log_scalar('test_sym_nx_recon_loss', info['sym_nx_recon_loss'])
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
                img_data = test_data[1]
                trj_data = test_data[2]
                snd_data = test_data[3]
                sym_data = torch.nn.functional.one_hot(test_data[0], num_classes=10).float()

                if self.model.use_cuda:
                    img_data = img_data.cuda()
                    trj_data = trj_data.cuda()
                    snd_data = snd_data.cuda()
                    sym_data = sym_data.cuda()

                # Generate modalities from complete nexus
                img_out, trj_out, snd_out, sym_out  = self.model.generate(x_img=img_data, x_trj=trj_data, x_snd=snd_data, x_sym=sym_data)
                img_out, img_nx_out = img_out[0], img_out[1]
                trj_out, trj_nx_out = trj_out[0], trj_out[1]
                snd_out, snd_nx_out = snd_out[0], snd_out[1]
                sym_out, sym_nx_out = sym_out[0], sym_out[1]

                # Generate modalities from single modality
                img_nx_img_out, img_nx_trj_out, img_nx_snd_out, img_nx_sym_out = self.model.generate(x_img=img_data)
                img_nx_img_out = img_nx_img_out[1]
                img_nx_trj_out = img_nx_trj_out[1]
                img_nx_snd_out = img_nx_snd_out[1]
                img_nx_sym_out = img_nx_sym_out[1]

                trj_nx_img_out, trj_nx_trj_out, trj_nx_snd_out, trj_nx_sym_out = self.model.generate(x_trj=trj_data)
                trj_nx_img_out = trj_nx_img_out[1]
                trj_nx_trj_out = trj_nx_trj_out[1]
                trj_nx_snd_out = trj_nx_snd_out[1]
                trj_nx_sym_out = trj_nx_sym_out[1]

                snd_nx_img_out, snd_nx_trj_out, snd_nx_snd_out, snd_nx_sym_out = self.model.generate(x_snd=snd_data)
                snd_nx_img_out = snd_nx_img_out[1]
                snd_nx_trj_out = snd_nx_trj_out[1]
                snd_nx_snd_out = snd_nx_snd_out[1]
                snd_nx_sym_out = snd_nx_sym_out[1]

                sym_nx_img_out, sym_nx_trj_out, sym_nx_snd_out, sym_nx_sym_out = self.model.generate(x_sym=sym_data)
                sym_nx_img_out = sym_nx_img_out[1]
                sym_nx_trj_out = sym_nx_trj_out[1]
                sym_nx_snd_out = sym_nx_snd_out[1]
                sym_nx_sym_out = sym_nx_sym_out[1]

                # Image Recon
                img_comp = torch.cat([img_data.view(-1, 1, 28, 28).cpu(), img_out.view(-1, 1, 28, 28).cpu()])
                all_nx_img_comp = torch.cat([img_data.view(-1, 1, 28, 28).cpu(), img_nx_out.view(-1, 1, 28, 28).cpu()])
                img_nx_img_comp = torch.cat([img_data.view(-1, 1, 28, 28).cpu(), img_nx_img_out.view(-1, 1, 28, 28).cpu()])
                trj_nx_img_comp = torch.cat(
                    [img_data.view(-1, 1, 28, 28).cpu(), trj_nx_img_out.view(-1, 1, 28, 28).cpu()])
                snd_nx_img_comp = torch.cat(
                    [img_data.view(-1, 1, 28, 28).cpu(), snd_nx_img_out.view(-1, 1, 28, 28).cpu()])
                sym_nx_img_comp = torch.cat([img_data.view(-1, 1, 28, 28).cpu(), sym_nx_img_out.view(-1, 1, 28, 28).cpu()])

                # Text Recon
                sym_res = np.argmax(F.log_softmax(sym_out, dim=-1).cpu().numpy(), axis=1).tolist()
                sym_res_str = ''
                for i, item in enumerate(sym_res):
                    sym_res_str += str(item) + " "

                sym_nx_res = np.argmax(F.log_softmax(sym_nx_out, dim=-1).cpu().numpy(), axis=1).tolist()
                sym_nx_res_str = ''
                for i, item in enumerate(sym_nx_res):
                    sym_nx_res_str += str(item) + " "

                img_nx_sym_out_res = np.argmax(F.log_softmax(img_nx_sym_out, dim=-1).cpu().numpy(), axis=1).tolist()
                img_nx_sym_out_res_str = ''
                for i, item in enumerate(img_nx_sym_out_res):
                    img_nx_sym_out_res_str += str(item) + " "

                trj_nx_sym_out_res = np.argmax(F.log_softmax(trj_nx_sym_out, dim=-1).cpu().numpy(), axis=1).tolist()
                trj_nx_sym_out_res_str = ''
                for i, item in enumerate(trj_nx_sym_out_res):
                    trj_nx_sym_out_res_str += str(item) + " "

                snd_nx_sym_out_res = np.argmax(F.log_softmax(snd_nx_sym_out, dim=-1).cpu().numpy(), axis=1).tolist()
                snd_nx_sym_out_res_str = ''
                for i, item in enumerate(snd_nx_sym_out_res):
                    snd_nx_sym_out_res_str += str(item) + " "

                sym_nx_sym_out_res = np.argmax(F.log_softmax(sym_nx_sym_out, dim=-1).cpu().numpy(), axis=1).tolist()
                sym_nx_sym_out_res_str = ''
                for i, item in enumerate(sym_nx_sym_out_res):
                    sym_nx_sym_out_res_str += str(item) + " "


            # Save data
            # Image
            t_utils.save_image(data=img_comp, path=results_dir, filename='img_mod_e' + str(epoch) + '.png',
                               data_size=img_data.size(0))
            ex.add_artifact(os.path.join(results_dir, "img_mod_e" + str(epoch) + '.png'),
                            name="image_recon_e" + str(epoch) + '.png')

            t_utils.save_image(data=all_nx_img_comp, path=results_dir, filename='all_nx_img_comp_e' + str(epoch) + '.png',
                               data_size=img_data.size(0))
            ex.add_artifact(os.path.join(results_dir, "all_nx_img_comp_e" + str(epoch) + '.png'),
                            name="all_nexus_image_comp_e" + str(epoch) + '.png')

            t_utils.save_image(data=img_nx_img_comp, path=results_dir, filename='img_nx_img_comp_e' + str(epoch) + '.png',
                               data_size=img_data.size(0))
            ex.add_artifact(os.path.join(results_dir, 'img_nx_img_comp_e' + str(epoch) + '.png'),
                            name="image_nexus_image_comp_e" + str(epoch) + '.png')

            t_utils.save_image(data=trj_nx_img_comp, path=results_dir,
                               filename='trj_nx_img_comp_e' + str(epoch) + '.png',
                               data_size=img_data.size(0))
            ex.add_artifact(os.path.join(results_dir, 'trj_nx_img_comp_e' + str(epoch) + '.png'),
                            name="trajectory_nexus_image_comp_e" + str(epoch) + '.png')

            t_utils.save_image(data=snd_nx_img_comp, path=results_dir,
                               filename='snd_nx_img_comp_e' + str(epoch) + '.png',
                               data_size=img_data.size(0))
            ex.add_artifact(os.path.join(results_dir, 'snd_nx_img_comp_e' + str(epoch) + '.png'),
                            name="sound_nexus_image_comp_e" + str(epoch) + '.png')

            t_utils.save_image(data=sym_nx_img_comp, path=results_dir,
                               filename='sym_nx_img_comp_e' + str(epoch) + '.png',
                               data_size=img_data.size(0))
            ex.add_artifact(os.path.join(results_dir, 'sym_nx_img_comp_e' + str(epoch) + '.png'),
                            name="symbol_nexus_image_comp_e" + str(epoch) + '.png')


            # Trajectory
            t_utils.save_trajectory(data=trj_data, path=results_dir, filename='trj_og_e' + str(epoch) + '.png',
                                    traj_norm=self.trj_norm,
                                    tmp_path=results_dir)
            ex.add_artifact(os.path.join(results_dir, "trj_og_e" + str(epoch) + '.png'),
                            name="trajectory_og_e" + str(epoch) + '.png')


            t_utils.save_trajectory(data=trj_out, path=results_dir, filename='trj_mod_e' + str(epoch) + '.png',
                                    traj_norm=self.trj_norm,
                                    tmp_path=results_dir)
            ex.add_artifact(os.path.join(results_dir, "trj_mod_e" + str(epoch) + '.png'),
                            name="trajectory_recon_e" + str(epoch) + '.png')

            t_utils.save_trajectory(data=trj_nx_out, path=results_dir, filename='all_nx_trj_e' + str(epoch) + '.png',
                                    traj_norm=self.trj_norm,
                                    tmp_path=results_dir)
            ex.add_artifact(os.path.join(results_dir, "all_nx_trj_e" + str(epoch) + '.png'),
                            name="all_nexus_trajectory_e" + str(epoch) + '.png')

            t_utils.save_trajectory(data=img_nx_trj_out, path=results_dir, filename='img_nx_trj_e' + str(epoch) + '.png',
                                    traj_norm=self.trj_norm,
                                    tmp_path=results_dir)
            ex.add_artifact(os.path.join(results_dir, "img_nx_trj_e" + str(epoch) + '.png'),
                            name="image_nexus_trajectory_e" + str(epoch) + '.png')

            t_utils.save_trajectory(data=trj_nx_trj_out, path=results_dir,
                                    filename='trj_nx_trj_e' + str(epoch) + '.png',
                                    traj_norm=self.trj_norm,
                                    tmp_path=results_dir)
            ex.add_artifact(os.path.join(results_dir, "trj_nx_trj_e" + str(epoch) + '.png'),
                            name="trajectory_nexus_trajectory_e" + str(epoch) + '.png')

            t_utils.save_trajectory(data=snd_nx_trj_out, path=results_dir,
                                    filename='snd_nx_trj_e' + str(epoch) + '.png',
                                    traj_norm=self.trj_norm,
                                    tmp_path=results_dir)
            ex.add_artifact(os.path.join(results_dir, "snd_nx_trj_e" + str(epoch) + '.png'),
                            name="sound_nexus_trajectory_e" + str(epoch) + '.png')

            t_utils.save_trajectory(data=sym_nx_trj_out, path=results_dir,
                                    filename='sym_nx_trj_e' + str(epoch) + '.png',
                                    traj_norm=self.trj_norm,
                                    tmp_path=results_dir)
            ex.add_artifact(os.path.join(results_dir, "sym_nx_trj_e" + str(epoch) + '.png'),
                            name="symbol_nexus_trajectory_e" + str(epoch) + '.png')

            # Sound
            t_utils.save_sound(data=snd_data, path=results_dir, filename='snd_og_e' + str(epoch),
                               sound_norm=self.snd_norm)
            for i in range(1):
                ex.add_artifact(os.path.join(results_dir, "snd_og_e" + str(epoch) + '_' + str(i) + '.wav'),
                                name="sound_og_e" + str(epoch) + '_' + str(i) + ".wav")

            t_utils.save_sound(data=snd_out, path=results_dir, filename='snd_mod_e' + str(epoch),
                               sound_norm=self.snd_norm)
            for i in range(1):
                ex.add_artifact(os.path.join(results_dir, "snd_mod_e" + str(epoch) + '_' + str(i) + '.wav'),
                                name="sound_recon_e" + str(epoch) + '_' + str(i) + ".wav")


            t_utils.save_sound(data=snd_nx_out, path=results_dir, filename='all_nx_snd_e' + str(epoch),
                               sound_norm=self.snd_norm)
            for i in range(1):
                ex.add_artifact(os.path.join(results_dir, "all_nx_snd_e" + str(epoch) + '_' + str(i) + '.wav'),
                                name="all_nexus_sound_e" + str(epoch) + '_' + str(i) + ".wav")


            t_utils.save_sound(data=img_nx_snd_out, path=results_dir, filename='img_nx_snd_e' + str(epoch),
                               sound_norm=self.snd_norm)
            for i in range(1):
                ex.add_artifact(os.path.join(results_dir, "img_nx_snd_e" + str(epoch) + '_' + str(i) + '.wav'),
                                name="image_nexus_sound_e" + str(epoch) + '_' + str(i) + ".wav")

            t_utils.save_sound(data=trj_nx_snd_out, path=results_dir, filename='trj_nx_snd_e' + str(epoch),
                               sound_norm=self.snd_norm)
            for i in range(1):
                ex.add_artifact(os.path.join(results_dir, "trj_nx_snd_e" + str(epoch) + '_' + str(i) + '.wav'),
                                name="trajectory_nexus_sound_e" + str(epoch) + '_' + str(i) + ".wav")

            t_utils.save_sound(data=snd_nx_snd_out, path=results_dir, filename='snd_nx_snd_e' + str(epoch),
                               sound_norm=self.snd_norm)
            for i in range(1):
                ex.add_artifact(os.path.join(results_dir, "snd_nx_snd_e" + str(epoch) + '_' + str(i) + '.wav'),
                                name="sound_nexus_sound_e" + str(epoch) + '_' + str(i) + ".wav")

            t_utils.save_sound(data=sym_nx_snd_out, path=results_dir, filename='sym_nx_snd_e' + str(epoch),
                               sound_norm=self.snd_norm)
            for i in range(1):
                ex.add_artifact(os.path.join(results_dir, "sym_nx_snd_e" + str(epoch) + '_' + str(i) + '.wav'),
                                name="symbol_nexus_sound_e" + str(epoch) + '_' + str(i) + ".wav")


            # Symbol
            with open(os.path.join(results_dir,'sym_res_str_e' + str(epoch) + '.txt'), "w") as text_file:
                print(sym_res_str, file=text_file)
            ex.add_artifact(os.path.join(results_dir, "sym_res_str_e" + str(epoch) + '.txt'),
                            name= "symbol_recon_e" + str(epoch) + '.txt')

            with open(os.path.join(results_dir,'sym_nx_res_str_e' + str(epoch) + '.txt'), "w") as text_file:
                print(sym_nx_res_str, file=text_file)
            ex.add_artifact(os.path.join(results_dir, "sym_nx_res_str_e" + str(epoch) + '.txt'),
                            name= "all_nexus_symbol_recon_e" + str(epoch) + '.txt')

            with open(os.path.join(results_dir,'img_nx_sym_out_res_str_e' + str(epoch) + '.txt'), "w") as text_file:
                print(img_nx_sym_out_res_str, file=text_file)
            ex.add_artifact(os.path.join(results_dir, "img_nx_sym_out_res_str_e" + str(epoch) + '.txt'),
                            name= "image_nexus_symbol_recon_e" + str(epoch) + '.txt')

            with open(os.path.join(results_dir, 'trj_nx_sym_out_res_str_e' + str(epoch) + '.txt'), "w") as text_file:
                print(trj_nx_sym_out_res_str, file=text_file)
            ex.add_artifact(os.path.join(results_dir, "trj_nx_sym_out_res_str_e" + str(epoch) + '.txt'),
                            name="trajectory_nexus_symbol_recon_e" + str(epoch) + '.txt')

            with open(os.path.join(results_dir, 'snd_nx_sym_out_res_str_e' + str(epoch) + '.txt'), "w") as text_file:
                print(snd_nx_sym_out_res_str, file=text_file)
            ex.add_artifact(os.path.join(results_dir, "snd_nx_sym_out_res_str_e" + str(epoch) + '.txt'),
                            name="sound_nexus_symbol_recon_e" + str(epoch) + '.txt')

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
            log_dir_path('trained_models'), 'nexus_checkpoint.pth.tar'),
        name='nexus_last_checkpoint.pth.tar')
    ex.add_artifact(
        os.path.join(
            log_dir_path('trained_models'), 'best_nexus_model.pth.tar'),
        name='best_nexus_model.pth.tar')


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
    nx_info, img_info, trj_info, snd_info, sym_info = t_utils.get_specs(model_config)
    # Load sound vae
    snd_vae = t_utils.load_sound_vae(gpu_config)

    model = NexusModel(nx_info=nx_info,
                        img_info=img_info,
                       trj_info=trj_info,
                       snd_info=snd_info,
                        sym_info=sym_info,
                       snd_vae=snd_vae,
                       use_cuda=gpu_config['cuda'])

    # Create trainer
    trainer = Trainer(model, training_config, gpu_config['cuda'])

    # Create Dataset
    dataset = MultimodalDataset(
        data_dir=exp_dir_path('../../../scenarios/multimodal_dataset/data'),
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