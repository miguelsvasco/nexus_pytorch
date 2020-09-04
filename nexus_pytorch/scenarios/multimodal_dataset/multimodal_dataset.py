import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
from nexus_pytorch.scenarios.multimodal_dataset.utils.sound_utils import unstack_tensor
import os


class MMDataset(Dataset):
    def __init__(self, data_file, modalities=None):

        self.data_file = data_file
        self.modalities = modalities
        if not os.path.exists(data_file):
                raise RuntimeError(
                    'Dataset not found. Please generate dataset and place it in the data folder.')

        self._s_data, self._i_data, self._t_data,\
        self._a_data, self._traj_normalization, self._audio_normalization = None, None, None, None, None, None

        if modalities is None:
            self._s_data, self._i_data, self._t_data, self._a_data, self._traj_normalization, self._audio_normalization = torch.load(data_file)

        else:
            if "image" in modalities:
                self._s_data, self._i_data, _, _, self._traj_normalization, self._audio_normalization = torch.load(data_file)
            if "trajectory" in modalities:
                self._s_data, _, self._t_data, _, self._traj_normalization, self._audio_normalization = torch.load(data_file)
            if "sound" in modalities:
                self._s_data, _, _, self._a_data, self._traj_normalization, self._audio_normalization = torch.load(data_file)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (t_data, m_data, f_data)
        """
        if self.modalities is None:

            # Audio modality is a 3x32x32 representation, need to unstack!
            audio = unstack_tensor(self._a_data[index]).unsqueeze(0)
            audio_perm = audio.permute(0, 2, 1)
            return self._s_data[index], self._i_data[index], self._t_data[index], audio_perm
        else:
            data = [self._s_data[index]]
            if self._i_data is not None:
                data.append(self._i_data[index])
            if self._t_data is not None:
                data.append(self._t_data[index])
            if self._a_data is not None:
                audio = unstack_tensor(self._a_data[index]).unsqueeze(0)
                audio_perm = audio.permute(0, 2, 1)
                data.append(audio_perm)
            return data


    def __len__(self):
        return len(self._s_data)

    def get_audio_normalization(self):
        return self._audio_normalization

    def get_traj_normalization(self):
        return self._traj_normalization



class MultimodalDataset():
    def __init__(self,
                 data_dir,
                 modalities=None,
                 batch_size=64,
                 eval_samples=10,
                 validation_size=0.1,
                 seed=42):

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.validation_size = validation_size
        self.eval_samples = eval_samples
        self.seed = seed
        self.modalities = modalities

        # Check which dataset to load
        self.train_data_file = "mhd_train.pt"
        self.test_data_file  = "mhd_test.pt"

        # Idx
        self.train_idx, self.validation_idx, self.test_idx = None, None, None
        self.audio_normalization = None
        self.traj_normalization = None

        # Data
        self.train_loader, self.val_loader = self.get_training_loaders()
        self.test_loader = self.get_test_loader()


    def get_training_loaders(self):

        train_dataset = MMDataset(data_file=os.path.join(self.data_dir, self.train_data_file),
                                  modalities=self.modalities)
        valid_dataset = MMDataset(data_file=os.path.join(self.data_dir, self.train_data_file),
                                  modalities=self.modalities)

        self.audio_normalization = train_dataset.get_audio_normalization()
        self.traj_normalization = train_dataset.get_traj_normalization()

        # Load train and val idx
        self.train_idx, self.validation_idx = self.get_train_val_idx(train_dataset=train_dataset)

        print("Training Dataset Samples = " + str(len(self.train_idx)))
        print("Validation Dataset Samples = " + str(len(self.validation_idx)))

        train_sampler = SubsetRandomSampler(self.train_idx)
        valid_sampler = SubsetRandomSampler(self.validation_idx)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            num_workers=1,
            pin_memory=True)

        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            sampler=valid_sampler,
            num_workers=1,
            pin_memory=True)

        return train_loader, valid_loader

    def get_test_loader(self, bsize=None):

        # load the dataset
        test_dataset = MMDataset(data_file=os.path.join(self.data_dir, self.test_data_file),
                                 modalities=self.modalities)

        if bsize is None:
            bsize = self.eval_samples

        return torch.utils.data.DataLoader(test_dataset,
                                           batch_size=bsize,
                                           num_workers=1,
                                           pin_memory=True)

    def get_train_val_idx(self, train_dataset):

        # Get global indices
        num_train = len(train_dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)

        # Split Training and validation
        split = int(np.floor(self.validation_size * num_train))

        # Training and validation indexes
        return indices[split:], indices[:split]

    def get_test_idx(self, test_dataset):

        # Get global indices
        num_test = len(test_dataset)
        indices = list(range(num_test))

        # Test indexes
        return indices


    def get_sound_normalization(self):
        return self.audio_normalization

    def get_traj_normalization(self):
        return self.traj_normalization
