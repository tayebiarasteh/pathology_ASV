"""
speaker_data_loader.py
Created on Nov 22, 2021.
Data loader.

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de>
https://github.com/tayebiarasteh/
"""

import numpy as np
import torch
import os
import pdb
import glob
import random
import pandas as pd
from torch.utils.data import Dataset

from config.serde import read_config


epsilon = 1e-15



class tisv_dataset_train_valid(Dataset):
    def __init__(self, cfg_path='./config/config.yaml', seed=1, training=True, experiment_name='name'):
        """text-independent speaker verification data loader for training
        and validation data for GE2E method.

        Parameters
        ----------
        cfg_path: str
            Config file path of the experiment

        seed: int
            seed for random generator

        training: bool
            if you want to generate training or validation data

        Returns
        -------
        output_tensor: torch tensor
            loaded mel spectrograms
        """

        self.params = read_config(cfg_path)
        self.cfg_path = cfg_path
        self.seed = seed
        self.file_path = self.params['file_path']
        self.training = training

        if training:
            self.sampling_val = np.random.randint(140, 180, 1)[0]
            self.main_df = pd.read_csv(os.path.join(self.params['file_path'], "tisv_preprocess/" + experiment_name + "/train_" + experiment_name + ".csv"), sep=';')
            self.M = self.params['Network']['M']

        else:
            self.sampling_val = self.params['preprocessing']['n_frame_test']
            self.main_df = pd.read_csv(os.path.join(self.params['file_path'], "tisv_preprocess/" + experiment_name + "/valid_" + experiment_name + ".csv"), sep=';')
            self.M = self.params['Network']['M_valid']

        self.speaker_list = self.main_df['speaker_id'].unique().tolist()



    def __len__(self):
        return len(self.speaker_list)

    def __getitem__(self, idx):

        output_tensor = []

        # select a speaker
        selected_speaker = self.speaker_list[idx]
        selected_speaker_df = self.main_df[self.main_df['speaker_id'] == selected_speaker]

        # randomly select M partial utterances from the speaker
        shuff_selected_speaker_df = selected_speaker_df.sample(frac=1).reset_index(drop=True)

        shuff_selected_speaker_df = shuff_selected_speaker_df[:self.M]

        # return M partial utterances
        for index, row in shuff_selected_speaker_df.iterrows():
            # select a random partial utterance
            utterance = np.load(os.path.join(self.file_path, row['relative_path']))
            # randomly sample a fixed specified length
            id = np.random.randint(0, utterance.shape[1] - self.sampling_val, 1)
            utterance = utterance[:, id[0]:id[0] + self.sampling_val]
            output_tensor.append(utterance)

        output_tensor = np.stack(output_tensor)
        output_tensor = torch.from_numpy(np.transpose(output_tensor, axes=(0, 2, 1)))
        # transpose [batch, frames=sampling_val, n_mels]

        return output_tensor


class tisv_dvector_creator_loader:
    def __init__(self, cfg_path='./config/config.json', experiment_name='name'):
        """For d-vector creation (prediction of the input utterances) step.

        Parameters
        ----------
        cfg_path: str
            Config file path of the experiment
        """
        params = read_config(cfg_path)
        self.cfg_path = cfg_path
        self.file_path = params['file_path']

        self.main_df = pd.read_csv(os.path.join(params['file_path'], "tisv_preprocess/" + experiment_name + "/test_" + experiment_name + ".csv"), sep=';')
        self.speaker_list = self.main_df['speaker_id'].unique().tolist()


    def provide_data(self):
        """
        Returns
        -------
        speakers: dictionary of list
            a dictionary of all the speakers. Each speaker contains a list of
            all its utterances converted to mel spectrograms
        """
        # dictionary of speakers
        speakers = {}

        for speaker_name in self.speaker_list:
            selected_speaker_df = self.main_df[self.main_df['speaker_id'] == speaker_name]
            # list of utterances of each speaker
            utterances = []

            for index, row in selected_speaker_df.iterrows():
                utterance = np.load(os.path.join(self.file_path, row['relative_path']))
                utterance = torch.from_numpy(np.transpose(utterance, axes=(1, 0)))
                utterances.append(utterance)
            speakers[speaker_name] = utterances

        return speakers



class tisv_after_dvector_loader:
    def __init__(self, cfg_path='./configs/config.json', M=12):
        """For thresholding and testing.

        Parameters
        ----------
        cfg_path: str
            Config file path of the experiment

        M: int
            number of partial utterances per speaker
            must be an even number

        Returns
        -------
        output_tensor: torch tensor
            loaded mel spectrograms
            return shape: (# all speakers, M, embedding size)
        """
        params = read_config(cfg_path)
        self.speaker_list = glob.glob(os.path.join(params['target_dir'], params['dvectors_path'], "*.npy"))
        self.M = M


    def provide_test(self):
        output_tensor = []

        # return all speakers
        for speaker in self.speaker_list:
            embedding = np.load(speaker)

            # randomly sample a fixed specified length
            if embedding.shape[0] == self.M:
                id = np.array([0])
            else:
                id = np.random.randint(0, embedding.shape[0] - self.M, 1)
            embedding = embedding[id[0]:id[0] + self.M]
            output_tensor.append(embedding)
        output_tensor = np.stack(output_tensor)
        output_tensor = torch.from_numpy(output_tensor)

        return output_tensor




class tisv_after_dvector_loader_forscattering:
    def __init__(self, cfg_path='./configs/config.json', M=12, experiment_name='name'):
        """For thresholding and testing.

        Parameters
        ----------
        cfg_path: str
            Config file path of the experiment

        M: int
            number of partial utterances per speaker
            must be an even number

        Returns
        -------
        output_tensor: torch tensor
            loaded mel spectrograms
            return shape: (# all speakers, M, embedding size)
        """
        params = read_config(cfg_path)
        self.speaker_list = glob.glob(os.path.join(params['target_dir'], params['dvectors_path'], "*.npy"))
        self.speaker_list.sort()
        self.main_df = pd.read_csv(os.path.join(params['file_path'], "tisv_preprocess/" + experiment_name + "/test_" + experiment_name + ".csv"), sep=';')
        self.M = M


    def provide_test(self):
        output_tensor = []
        output_WR_list = []
        output_WA_list = []
        speaker_name_list = []
        diagnosis_list = []
        age_list = []
        intelligibility_list = []
        user_id_list = []
        mic_room_list = []
        gender_list = []

        # return all speakers
        for speaker in self.speaker_list:
            speaker_name = os.path.basename(speaker).split(".")[0]
            speaker_name_list.append(speaker_name)
            try:
                output_WR_list.append(self.main_df[self.main_df['speaker_id']==speaker_name]['WR'].values[0])
            except:
                output_WR_list.append(-1000)
            diagnosis_list.append(self.main_df[self.main_df['speaker_id']==speaker_name]['diagnosis'].values[0])
            age_list.append(self.main_df[self.main_df['speaker_id']==speaker_name]['age'].values[0])
            try:
                output_WA_list.append(self.main_df[self.main_df['speaker_id']==speaker_name]['WA'].values[0])
            except:
                output_WA_list.append(-1000)
            try:
                intelligibility_list.append(self.main_df[self.main_df['speaker_id']==speaker_name]['intelligibility'].values[0])
            except:
                intelligibility_list.append(-1000)
            user_id_list.append(self.main_df[self.main_df['speaker_id']==speaker_name]['user_id'].values[0])
            mic_room_list.append(self.main_df[self.main_df['speaker_id']==speaker_name]['mic_room'].values[0])
            gender_list.append(self.main_df[self.main_df['speaker_id']==speaker_name]['gender'].values[0])
            embedding = np.load(speaker)

            # randomly sample a fixed specified length
            if embedding.shape[0] == self.M:
                id = np.array([0])
            else:
                id = np.random.randint(0, embedding.shape[0] - self.M, 1)
            embedding = embedding[id[0]:id[0] + self.M]
            output_tensor.append(embedding)
        output_tensor = np.stack(output_tensor)
        output_tensor = torch.from_numpy(output_tensor)

        output_WR_list = np.stack(output_WR_list, 0)
        return output_tensor, output_WR_list, speaker_name_list, diagnosis_list, age_list, output_WA_list, intelligibility_list, user_id_list, mic_room_list, gender_list



