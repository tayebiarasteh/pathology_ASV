"""
PEAKS_specific_data_preprocess.py
Created on Jan 20, 2022.
Data preprocessing for text independent speaker verification for PEAKS only.

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de>
https://github.com/tayebiarasteh/
"""

import glob
import os
import pdb
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import webrtcvad
import struct
from tqdm import tqdm
import random
from math import ceil, isnan
from scipy.ndimage.morphology import binary_dilation
import matplotlib.pyplot as plt

from config.serde import read_config

import warnings
warnings.filterwarnings('ignore')


# Global variables
int16_max = (2 ** 15) - 1



class data_preprocess_PEAKS():
    def __init__(self, cfg_path="/PATH/config/config.yaml"):
        self.params = read_config(cfg_path)


    def histogram_age(self, mic_room='maxillofacial', test_type='PLAKSS_BILDTEXT', lowerbound=0, upperbound=100):

        file_path_input = "/PATH.csv"
        df_peaks = pd.read_csv(file_path_input, sep=';')

        selected_df = df_peaks[df_peaks['test_type'] == test_type]
        # selected_df = selected_df[selected_df['mic_room'] == mic_room]
        selected_df1 = selected_df[selected_df['mic_room'] == 'maxillofacial']
        selected_df2 = selected_df[selected_df['mic_room'] == 'plantronics']
        # selected_df3 = selected_df[selected_df['mic_room'] == 'logitech']
        selected_df = selected_df1.append(selected_df2)
        # selected_df = selected_df.append(selected_df3)

        selected_df = selected_df[selected_df['age'] > 0]
        selected_df = selected_df[selected_df['age'] < 20.5]
        # selected_df = selected_df[selected_df['WR'] > 0]

        # if we want to remove some ages for statistics use below
        ##############################
        # initiating the df
        final_data = pd.DataFrame(columns=['relative_path', 'speaker_id', 'test_type', 'session', 'file_length', 'user_id', 'mic_room', 'age',
                                                 'WA', 'WR', 'intelligibility', 'diagnosis', 'surgery', 'gender', 'father_tongue', 'mother_tongue', 'chemo_therapy', 'dental_prosthesis', 'irradiation'])

        PEAKS_speaker_list = selected_df['speaker_id'].unique().tolist()
        age_speaker_df = pd.DataFrame(columns=['speaker_id', 'age'])

        for speaker in PEAKS_speaker_list:
            tempp = pd.DataFrame([[speaker, selected_df[selected_df['speaker_id'] == speaker]['age'].mean()]],
                                 columns=['speaker_id', 'age'])
            age_speaker_df = age_speaker_df.append(tempp)

        age_speaker_df = age_speaker_df[age_speaker_df['age'] <= upperbound]
        age_speaker_df = age_speaker_df[age_speaker_df['age'] >= lowerbound]

        chosen_speaker_list = age_speaker_df['speaker_id'].unique().tolist()

        # adding files based on chosen ages
        for speaker in chosen_speaker_list:
            selected_speaker_df = selected_df[selected_df['speaker_id'] == speaker]
            final_data = final_data.append(selected_speaker_df)

        selected_df = final_data
        ################################


        PEAKS_speaker_list = selected_df['speaker_id'].unique().tolist()
        age_list = []
        for speaker in PEAKS_speaker_list:
            age_list.append(selected_df[selected_df['speaker_id'] == speaker]['age'].mean())

        plt.subplot(121)
        plt.hist(age_list)
        plt.xlabel('Age [years]')
        plt.ylabel('Num Speakers')
        age_list.sort()
        print(age_list, '\n')
        age_list = np.array(age_list)
        org_mean = np.mean(age_list)
        org_std = np.std(age_list)
        org_median = np.median(age_list)

        plt.title("full | " + mic_room + " | " + test_type + " | spk: " + str(len(PEAKS_speaker_list)) +
                  " | age: " + f"{org_mean:.1f}" + " +- " + f"{org_std:.1f}")

        age_list = age_list[age_list <= upperbound]
        age_list = age_list[age_list >= lowerbound]

        org_mean = np.mean(age_list)
        org_std = np.std(age_list)
        org_median = np.median(age_list)

        print('number of speakers:', len(selected_df['speaker_id'].unique()))
        print('number of utterances:', len(selected_df))
        print('Total utterance length in hours:', f"{selected_df['file_length'].sum() / 3600:.2f}")
        print('Mean age:', f"{org_mean:.2f}" + " +- " + f"{org_std:.2f}")
        print('median age:', org_median)


        plt.subplot(122)
        plt.hist(age_list)
        plt.xlabel('Age [years]')
        plt.ylabel('Num Speakers')
        plt.title("chosen | " + mic_room + " | " + test_type + " | spk: " + str(len(age_list)) +
                  " | age: " + f"{np.mean(age_list):.1f}" + " +- " + f"{np.std(age_list):.1f}")

        print("\ntotal number of chosen speakers:", len(age_list))
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()

        WR_mean = selected_df[selected_df['WR'] > 0]['WR'].mean()
        WR_std = selected_df[selected_df['WR'] > 0]['WR'].std()
        num_female_spk = len(selected_df[selected_df['gender'] == 'f']['speaker_id'].unique())
        num_male_spk = len(selected_df[selected_df['gender'] == 'm']['speaker_id'].unique())

        print('Mean WRR:', f"{WR_mean:.2f}" + " +- " + f"{WR_std:.2f}")
        print('number of female speakers:', num_female_spk)
        print('number of male speakers:', num_male_spk)




    def main(self, file_path_input ="/PATH.csv",
             mic_room='maxillofacial', test_type='PLAKSS_BILDTEXT', ratio=0.05, max_speakers=50,
             exp_name='maxillofacial_plakss', three_division=True, age=False, target_mean=0, target_std=0, lowerbound=0, upperbound=100):
        """main file after having a master csv, which divides to train, valid, test; and does preprocessing

        Parameters
        ----------
        mic_room: str
            name of the "mic_room" we want to choose. It should be exactly the same characters as in the spreadsheet.

        test_type: str
            name of the "test_type" we want to choose. It should be exactly the same characters as in the spreadsheet.

        ratio: float
            ratio of the split between train and (valid) and test.

        exp_name: str
            name of the experiment

        max_speakers: int
            maximum number of the speakers that we want to choose from the subset to work with.
            This number is the total sum of all the train, (valid), test subsets.

        three_division: bool
            if we want to split the data to train, valid, test or only to train and test, without valid.

        age: bool
            if we want to choose speakers based age distribution.

        lowerbound: float
            lower bound of the age we want to choose.

        upperbound: float
            upper bound of the age we want to choose.
        """

        train_output_df_path = os.path.join(self.params['file_path'], 'tisv_preprocess', exp_name, 'train_' + exp_name + '.csv')
        valid_output_df_path = os.path.join(self.params['file_path'], 'tisv_preprocess', exp_name, 'valid_' + exp_name + '.csv')
        test_output_df_path = os.path.join(self.params['file_path'], 'tisv_preprocess', exp_name, 'test_' + exp_name + '.csv')

        statistics_df_path = os.path.join(self.params['file_path'], 'tisv_preprocess', exp_name, 'statistics_' + exp_name + '.csv')
        histogram_path = os.path.join(self.params['file_path'], 'tisv_preprocess', exp_name, 'histogram_' + exp_name + '.png')

        df_peaks = pd.read_csv(file_path_input, sep=';')

        df_peaks = df_peaks[df_peaks['mic_room'] == mic_room]
        selected_df = df_peaks[df_peaks['test_type'] == test_type]

        # reducing the number of speakers
        # based on age
        if age:
            # selected_df = self.csv_size_reducer_fixed_age_based(selected_df, max_speakers, lowerbound, upperbound)
            selected_df = self.csv_size_reducer_std_mean_age_based(selected_df, max_speakers, target_mean, target_std)
        # at random
        else:
            selected_df = self.csv_size_reducer_random(selected_df, max_speakers)

        if three_division:
            # creating train, valid, test csv files
            final_train_df, final_valid_df, final_test_df = self.train_valid_test_csv_creator_PEAKS(selected_df, ratio)
        else:
            # creating train and test csv files
            final_train_df, final_test_df = self.train_test_csv_creator_PEAKS(selected_df, ratio)

        # tisv preprocessing
        print('\ntisv preprocess for training data\n')
        self.tisv_preproc_train(input_df=final_train_df, output_df_path=train_output_df_path, exp_name=exp_name)
        if three_division:
            print('\ntisv preprocess for validation data\n')
            self.tisv_preproc_test(input_df=final_valid_df, output_df_path=valid_output_df_path, exp_name=exp_name)
        print('\ntisv preprocess for test data\n')
        self.tisv_preproc_test(input_df=final_test_df, output_df_path=test_output_df_path, exp_name=exp_name)

        # saving histogram of the ages
        # train
        final_train_speaker_list = final_train_df['speaker_id'].unique().tolist()
        age_list_train = []
        for speaker in final_train_speaker_list:
            age_list_train.append(final_train_df[final_train_df['speaker_id'] == speaker]['age'].mean())
        plt.subplot(121)
        plt.hist(age_list_train)
        plt.xlabel('Age [years]')
        plt.ylabel('Num Speakers')
        plt.title("train | age: " + f"{np.mean(age_list_train):.1f}" + " +- " + f"{np.std(age_list_train):.1f}")

        # test
        final_test_speaker_list = final_test_df['speaker_id'].unique().tolist()
        age_list_test = []
        for speaker in final_test_speaker_list:
            age_list_test.append(final_test_df[final_test_df['speaker_id'] == speaker]['age'].mean())
        plt.subplot(122)
        plt.hist(age_list_test)
        plt.xlabel('Age [years]')
        plt.ylabel('Num Speakers')
        plt.title("test | age: " + f"{np.mean(age_list_test):.1f}" + " +- " + f"{np.std(age_list_test):.1f}")

        # plt.savefig(histogram_path)

        # statistics of the chosen dataset
        train_speaker_size = len(final_train_df['speaker_id'].unique().tolist())
        test_speaker_size = len(final_test_df['speaker_id'].unique().tolist())
        train_length = final_train_df['file_length'].sum() / 3600
        test_length = final_test_df['file_length'].sum() / 3600
        train_age_mean = final_train_df['age'].mean()
        test_age_mean = final_test_df['age'].mean()
        train_age_std = final_train_df['age'].std()
        test_age_std = final_test_df['age'].std()

        train_WA_mean = final_train_df[final_train_df['WA'] > -9999]['WA'].mean()
        train_WA_std = final_train_df[final_train_df['WA'] > -9999]['WA'].std()
        train_WR_mean = final_train_df[final_train_df['WR'] > 0]['WR'].mean()
        train_WR_std = final_train_df[final_train_df['WR'] > 0]['WR'].std()

        test_WA_mean = final_test_df[final_test_df['WA'] > -9999]['WA'].mean()
        test_WA_std = final_test_df[final_test_df['WA'] > -9999]['WA'].std()
        test_WR_mean = final_test_df[final_test_df['WR'] > 0]['WR'].mean()
        test_WR_std = final_test_df[final_test_df['WR'] > 0]['WR'].std()

        statistics_df = pd.DataFrame(columns=['subset', 'exp_name', 'total_speakers', 'total_hours', 'age_mean', 'age_std', 'WA_mean', 'WA_std', 'WR_mean', 'WR_std'])

        statistics_df = statistics_df.append(pd.DataFrame([['train', exp_name, train_speaker_size, train_length, train_age_mean, train_age_std,
                                                            train_WA_mean, train_WA_std, train_WR_mean, train_WR_std]],
                     columns=['subset', 'exp_name', 'total_speakers', 'total_hours', 'age_mean', 'age_std', 'WA_mean', 'WA_std', 'WR_mean', 'WR_std']))
        statistics_df = statistics_df.append(pd.DataFrame([['test', exp_name, test_speaker_size, test_length, test_age_mean, test_age_std,
                                                            test_WA_mean, test_WA_std, test_WR_mean, test_WR_std]],
                     columns=['subset', 'exp_name', 'total_speakers', 'total_hours', 'age_mean', 'age_std', 'WA_mean', 'WA_std', 'WR_mean', 'WR_std']))
        if three_division:
            valid_speaker_size = len(final_valid_df['speaker_id'].unique().tolist())
            valid_length = final_valid_df['file_length'].sum() / 3600
            valid_age_mean = final_valid_df['age'].mean()
            valid_age_std = final_valid_df['age'].std()

            valid_WA_mean = final_valid_df[final_valid_df['WA'] > -9999]['WA'].mean()
            valid_WA_std = final_valid_df[final_valid_df['WA'] > -9999]['WA'].std()
            valid_WR_mean = final_valid_df[final_valid_df['WR'] > 0]['WR'].mean()
            valid_WR_std = final_valid_df[final_valid_df['WR'] > 0]['WR'].std()

            statistics_df = statistics_df.append(pd.DataFrame([['valid', exp_name, valid_speaker_size, valid_length, valid_age_mean, valid_age_std,
                                                                valid_WA_mean, valid_WA_std, valid_WR_mean, valid_WR_std]],
                             columns=['subset', 'exp_name', 'total_speakers', 'total_hours', 'age_mean', 'age_std', 'WA_mean', 'WA_std', 'WR_mean', 'WR_std']))

        statistics_df.to_csv(statistics_df_path, sep=';', index=False)



    def csv_age_matcher(self):
        """adding age from meta data to the main csv

        Parameters
        ----------
        """
        input_file_path = "/PATH.csv"
        input_df = pd.read_csv(input_file_path, sep=';')

        # PLAKSS
        input_subject_folder_path= "/PATH/"
        list_users = os.listdir(input_subject_folder_path)

        for user in tqdm(list_users):
            input_subject_path = os.path.join(input_subject_folder_path, user)
            input_subject_df = pd.read_csv(input_subject_path, sep=';')
            list_speakers_subject = input_subject_df['ID:'].unique().tolist()

            for speaker in list_speakers_subject:
                date_birth = input_subject_df[input_subject_df['ID:'] == speaker]['Date of birth:'].values[0]
                date_record = input_subject_df[input_subject_df['ID:'] == speaker]['Record'].values[0]
                input_df.loc[input_df.speaker_id == str(speaker), 'age'] = self.date_to_age(
                    os.path.basename(date_birth).split(" ")[0], os.path.basename(date_record).split(" ")[0])

        # NORDWIND
        input_subject_folder_path= "/PATH/"

        list_users = os.listdir(input_subject_folder_path)

        for user in tqdm(list_users):
            input_subject_path = os.path.join(input_subject_folder_path, user)
            input_subject_df = pd.read_csv(input_subject_path, sep=';')
            list_speakers_subject = input_subject_df['ID:'].unique().tolist()

            for speaker in list_speakers_subject:
                date_birth = input_subject_df[input_subject_df['ID:'] == speaker]['Date of birth:'].values[0]
                date_record = input_subject_df[input_subject_df['ID:'] == speaker]['Record'].values[0]
                input_df.loc[input_df.speaker_id == str(speaker), 'age'] = self.date_to_age(
                    os.path.basename(date_birth).split(" ")[0], os.path.basename(date_record).split(" ")[0])

        input_df = input_df.fillna(-1)
        input_df.to_csv(input_file_path, sep=';', index=False)



    def csv_WR_adder(self):
        """adding WR, diagnostic, and surgery type from meta data to the main csv

        Parameters
        ----------
        """
        input_file_path = "/PATH.csv"
        input_df = pd.read_csv(input_file_path, sep=';')

        # PLAKSS
        input_subject_folder_path= "/PATH/"
        list_users = os.listdir(input_subject_folder_path)

        for user in tqdm(list_users):
            input_subject_path = os.path.join(input_subject_folder_path, user)
            input_subject_df = pd.read_csv(input_subject_path, sep=';')
            list_speakers_subject = input_subject_df['ID:'].unique().tolist()

            for speaker in list_speakers_subject:
                WA = input_subject_df[input_subject_df['ID:'] == speaker]['automatische WA'].values[0]
                if not isinstance(WA, str):
                    if isnan(WA):
                        WA = -10000.0
                    if WA == 0:
                        WA = -10000.0
                if isinstance(WA, str):
                    WA = -10000.0
                input_df.loc[input_df.speaker_id == str(speaker), 'WA'] = WA

                WR = input_subject_df[input_subject_df['ID:'] == speaker][' automatische WR'].values[0]
                if not isinstance(WR, str):
                    if isnan(WR):
                        WR = -10000.0
                    if WR == 0:
                        WR = -10000.0
                if isinstance(WR, str):
                    WR = -10000.0
                input_df.loc[input_df.speaker_id == str(speaker), 'WR'] = WR

                if 'Verständlichkeit' in input_subject_df[input_subject_df['ID:'] == speaker]:
                    intelligibility = input_subject_df[input_subject_df['ID:'] == speaker]['Verständlichkeit'].values[0]
                    if isnan(intelligibility):
                        intelligibility = -10000.0
                else:
                    intelligibility = -10000.0
                input_df.loc[input_df.speaker_id == str(speaker), 'intelligibility'] = intelligibility

                if 'Diagnostic findings:' in input_subject_df[input_subject_df['ID:'] == speaker]:
                    diagnosis = input_subject_df[input_subject_df['ID:'] == speaker]['Diagnostic findings:'].values[0]
                    if not isinstance(diagnosis, str):
                        if isnan(diagnosis) and not isinstance(diagnosis, str):
                            diagnosis = 'NA'
                else:
                    diagnosis = 'NA'
                input_df.loc[input_df.speaker_id == str(speaker), 'diagnosis'] = diagnosis

                if 'Type of surgery:' in input_subject_df[input_subject_df['ID:'] == speaker]:
                    surgery = input_subject_df[input_subject_df['ID:'] == speaker]['Type of surgery:'].values[0]
                    if not isinstance(surgery, str):
                        if isnan(surgery):
                            surgery = 'NA'
                else:
                    surgery = 'NA'
                input_df.loc[input_df.speaker_id == str(speaker), 'surgery'] = surgery

                if 'Gender' in input_subject_df[input_subject_df['ID:'] == speaker]:
                    gender = input_subject_df[input_subject_df['ID:'] == speaker]['Gender'].values[0]
                    if not isinstance(gender, str):
                        if isnan(gender):
                            gender = 'NA'
                else:
                    gender = 'NA'
                input_df.loc[input_df.speaker_id == str(speaker), 'gender'] = gender

                if 'Father\'s tongue:' in input_subject_df[input_subject_df['ID:'] == speaker]:
                    father_tongue = input_subject_df[input_subject_df['ID:'] == speaker]['Father\'s tongue:'].values[0]
                    if not isinstance(father_tongue, str):
                        if isnan(father_tongue):
                            father_tongue = 'NA'
                else:
                    father_tongue = 'NA'
                input_df.loc[input_df.speaker_id == str(speaker), 'father_tongue'] = father_tongue

                if 'Mother\'s tongue:' in input_subject_df[input_subject_df['ID:'] == speaker]:
                    mother_tongue = input_subject_df[input_subject_df['ID:'] == speaker]['Mother\'s tongue:'].values[0]
                    if not isinstance(mother_tongue, str):
                        if isnan(mother_tongue):
                            mother_tongue = 'NA'
                else:
                    mother_tongue = 'NA'
                input_df.loc[input_df.speaker_id == str(speaker), 'mother_tongue'] = mother_tongue

                if 'Chemo therapy:' in input_subject_df[input_subject_df['ID:'] == speaker]:
                    chemo_therapy = input_subject_df[input_subject_df['ID:'] == speaker]['Chemo therapy:'].values[0]
                    if not isinstance(chemo_therapy, bool):
                        if isnan(chemo_therapy):
                            chemo_therapy = 'NA'
                else:
                    chemo_therapy = 'NA'
                input_df.loc[input_df.speaker_id == str(speaker), 'chemo_therapy'] = chemo_therapy

                if 'Dental prosthesis:' in input_subject_df[input_subject_df['ID:'] == speaker]:
                    dental_prosthesis = input_subject_df[input_subject_df['ID:'] == speaker]['Dental prosthesis:'].values[0]
                    if not isinstance(dental_prosthesis, bool):
                        if isnan(dental_prosthesis):
                            dental_prosthesis = 'NA'
                else:
                    dental_prosthesis = 'NA'
                input_df.loc[input_df.speaker_id == str(speaker), 'dental_prosthesis'] = dental_prosthesis

                if 'Irradiation:' in input_subject_df[input_subject_df['ID:'] == speaker]:
                    irradiation = input_subject_df[input_subject_df['ID:'] == speaker]['Irradiation:'].values[0]
                    if not isinstance(irradiation, bool):
                        if isnan(irradiation):
                            irradiation = 'NA'
                else:
                    irradiation = 'NA'
                input_df.loc[input_df.speaker_id == str(speaker), 'irradiation'] = irradiation


        # NORDWIND
        input_subject_folder_path= "/PATH/"

        list_users = os.listdir(input_subject_folder_path)

        for user in tqdm(list_users):
            input_subject_path = os.path.join(input_subject_folder_path, user)
            input_subject_df = pd.read_csv(input_subject_path, sep=';')
            list_speakers_subject = input_subject_df['ID:'].unique().tolist()

            for speaker in list_speakers_subject:
                WA = input_subject_df[input_subject_df['ID:'] == speaker]['automatische WA'].values[0]
                if not isinstance(WA, str):
                    if isnan(WA):
                        WA = -10000.0
                    if WA == 0:
                        WA = -10000.0
                input_df.loc[input_df.speaker_id == str(speaker), 'WA'] = WA

                WR = input_subject_df[input_subject_df['ID:'] == speaker][' automatische WR'].values[0]
                if not isinstance(WR, str):
                    if isnan(WR):
                        WR = -10000.0
                    if WR == 0:
                        WR = -10000.0
                input_df.loc[input_df.speaker_id == str(speaker), 'WR'] = WR

                if 'Verständlichkeit' in input_subject_df[input_subject_df['ID:'] == speaker]:
                    intelligibility = input_subject_df[input_subject_df['ID:'] == speaker]['Verständlichkeit'].values[0]
                    if isnan(intelligibility):
                        intelligibility = -10000.0
                else:
                    intelligibility = -10000.0
                input_df.loc[input_df.speaker_id == str(speaker), 'intelligibility'] = intelligibility

                if 'Diagnostic findings:' in input_subject_df[input_subject_df['ID:'] == speaker]:
                    diagnosis = input_subject_df[input_subject_df['ID:'] == speaker]['Diagnostic findings:'].values[0]
                    if not isinstance(diagnosis, str):
                        if isnan(diagnosis) and not isinstance(diagnosis, str):
                            diagnosis = 'NA'
                else:
                    diagnosis = 'NA'
                input_df.loc[input_df.speaker_id == str(speaker), 'diagnosis'] = diagnosis

                if 'Type of surgery:' in input_subject_df[input_subject_df['ID:'] == speaker]:
                    surgery = input_subject_df[input_subject_df['ID:'] == speaker]['Type of surgery:'].values[0]
                    if not isinstance(surgery, str):
                        if isnan(surgery):
                            surgery = 'NA'
                else:
                    surgery = 'NA'
                input_df.loc[input_df.speaker_id == str(speaker), 'surgery'] = surgery

                if 'Gender' in input_subject_df[input_subject_df['ID:'] == speaker]:
                    gender = input_subject_df[input_subject_df['ID:'] == speaker]['Gender'].values[0]
                    if not isinstance(gender, str):
                        if isnan(gender):
                            gender = 'NA'
                else:
                    gender = 'NA'
                input_df.loc[input_df.speaker_id == str(speaker), 'gender'] = gender

                if 'Father\'s tongue:' in input_subject_df[input_subject_df['ID:'] == speaker]:
                    father_tongue = input_subject_df[input_subject_df['ID:'] == speaker]['Father\'s tongue:'].values[0]
                    if not isinstance(father_tongue, str):
                        if isnan(father_tongue):
                            father_tongue = 'NA'
                else:
                    father_tongue = 'NA'
                input_df.loc[input_df.speaker_id == str(speaker), 'father_tongue'] = father_tongue

                if 'Mother\'s tongue:' in input_subject_df[input_subject_df['ID:'] == speaker]:
                    mother_tongue = input_subject_df[input_subject_df['ID:'] == speaker]['Mother\'s tongue:'].values[0]
                    if not isinstance(mother_tongue, str):
                        if isnan(mother_tongue):
                            mother_tongue = 'NA'
                else:
                    mother_tongue = 'NA'
                input_df.loc[input_df.speaker_id == str(speaker), 'mother_tongue'] = mother_tongue

                if 'Chemo therapy:' in input_subject_df[input_subject_df['ID:'] == speaker]:
                    chemo_therapy = input_subject_df[input_subject_df['ID:'] == speaker]['Chemo therapy:'].values[0]
                    if not isinstance(chemo_therapy, bool):
                        if isnan(chemo_therapy):
                            chemo_therapy = 'NA'
                else:
                    chemo_therapy = 'NA'
                input_df.loc[input_df.speaker_id == str(speaker), 'chemo_therapy'] = chemo_therapy

                if 'Dental prosthesis:' in input_subject_df[input_subject_df['ID:'] == speaker]:
                    dental_prosthesis = input_subject_df[input_subject_df['ID:'] == speaker]['Dental prosthesis:'].values[0]
                    if not isinstance(dental_prosthesis, bool):
                        if isnan(dental_prosthesis):
                            dental_prosthesis = 'NA'
                else:
                    dental_prosthesis = 'NA'
                input_df.loc[input_df.speaker_id == str(speaker), 'dental_prosthesis'] = dental_prosthesis

                if 'Irradiation:' in input_subject_df[input_subject_df['ID:'] == speaker]:
                    irradiation = input_subject_df[input_subject_df['ID:'] == speaker]['Irradiation:'].values[0]
                    if not isinstance(irradiation, bool):
                        if isnan(irradiation):
                            irradiation = 'NA'
                else:
                    irradiation = 'NA'
                input_df.loc[input_df.speaker_id == str(speaker), 'irradiation'] = irradiation

        input_df.to_csv(input_file_path, sep=';', index=False)



    def date_to_age(self, date_birth, date_record):
        """
        gives age in years
        """

        day_birth = int(os.path.basename(date_birth).split(".")[0])
        month_birth = int(os.path.basename(date_birth).split(".")[1])
        year_birth = int(os.path.basename(date_birth).split(".")[2])

        day_record = int(os.path.basename(date_record).split(".")[0])
        month_record = int(os.path.basename(date_record).split(".")[1])
        year_record = int(os.path.basename(date_record).split(".")[2])

        diff_d = day_record - day_birth
        diff_m = month_record - month_birth
        diff_y = year_record - year_birth

        diff_d /= 360
        diff_m /= 12
        age = diff_y + diff_m + diff_d
        return age



    def train_valid_test_csv_creator_PEAKS(self, input_df, ratio):
        """splits the csv file to train, valid, and test csv files
        Parameters
        ----------
        """
        # initiating valid and train dfs
        final_train_data = pd.DataFrame(columns=['relative_path', 'speaker_id', 'test_type', 'session', 'file_length', 'user_id', 'mic_room',
                                                 'age', 'WA', 'WR', 'intelligibility', 'diagnosis', 'surgery', 'gender', 'father_tongue', 'mother_tongue', 'chemo_therapy', 'dental_prosthesis', 'irradiation'])
        final_valid_data = pd.DataFrame(columns=['relative_path', 'speaker_id', 'test_type', 'session', 'file_length', 'user_id', 'mic_room',
                                                 'age', 'WA', 'WR', 'intelligibility', 'diagnosis', 'surgery', 'gender', 'father_tongue', 'mother_tongue', 'chemo_therapy', 'dental_prosthesis', 'irradiation'])
        final_test_data = pd.DataFrame(columns=['relative_path', 'speaker_id', 'test_type', 'session', 'file_length', 'user_id', 'mic_room',
                                                'age', 'WA', 'WR', 'intelligibility', 'diagnosis', 'surgery', 'gender', 'father_tongue', 'mother_tongue', 'chemo_therapy', 'dental_prosthesis', 'irradiation'])

        PEAKS_speaker_list = input_df['speaker_id'].unique().tolist()
        random.shuffle(PEAKS_speaker_list)
        val_num = ceil(len(PEAKS_speaker_list) / (1/ratio))

        # take X% of PEAKS speakers as validation
        val_speakers = PEAKS_speaker_list[:val_num]
        # take X% of PEAKS speakers as test
        test_speakers = PEAKS_speaker_list[val_num:2*val_num]
        # take rest of PEAKS speakers as training
        train_speakers = PEAKS_speaker_list[2*val_num:]

        # adding PEAKS files to valid
        for speaker in val_speakers:
            selected_speaker_df = input_df[input_df['speaker_id'] == speaker]
            final_valid_data = final_valid_data.append(selected_speaker_df)

        # adding PEAKS files to test
        for speaker in test_speakers:
            selected_speaker_df = input_df[input_df['speaker_id'] == speaker]
            final_test_data = final_test_data.append(selected_speaker_df)

        # adding PEAKS files to train
        for speaker in train_speakers:
            selected_speaker_df = input_df[input_df['speaker_id'] == speaker]
            final_train_data = final_train_data.append(selected_speaker_df)

        # sort based on speaker id
        final_train_data = final_train_data.sort_values(['relative_path'])
        final_valid_data = final_valid_data.sort_values(['relative_path'])
        final_test_data = final_test_data.sort_values(['relative_path'])

        return final_train_data, final_valid_data, final_test_data



    def train_test_csv_creator_PEAKS(self, input_df, ratio):
        """splits the csv file to train, valid, and test csv files
        Parameters
        ----------
        """
        # initiating valid and train dfs
        final_train_data = pd.DataFrame(columns=['relative_path', 'speaker_id', 'test_type', 'session', 'file_length', 'user_id', 'mic_room', 'age',
                                                 'WA', 'WR', 'intelligibility', 'diagnosis', 'surgery', 'gender', 'father_tongue', 'mother_tongue', 'chemo_therapy', 'dental_prosthesis', 'irradiation'])
        final_test_data = pd.DataFrame(columns=['relative_path', 'speaker_id', 'test_type', 'session', 'file_length', 'user_id', 'mic_room', 'age',
                                                'WA', 'WR', 'intelligibility', 'diagnosis', 'surgery', 'gender', 'father_tongue', 'mother_tongue', 'chemo_therapy', 'dental_prosthesis', 'irradiation'])

        PEAKS_speaker_list = input_df['speaker_id'].unique().tolist()
        random.shuffle(PEAKS_speaker_list)
        test_num = ceil(len(PEAKS_speaker_list) / (1/ratio))

        # take X% of PEAKS speakers as test
        test_speakers = PEAKS_speaker_list[:test_num]
        # take rest of PEAKS speakers as training
        train_speakers = PEAKS_speaker_list[test_num:]

        # adding PEAKS files to test
        for speaker in test_speakers:
            selected_speaker_df = input_df[input_df['speaker_id'] == speaker]
            final_test_data = final_test_data.append(selected_speaker_df)

        # adding PEAKS files to train
        for speaker in train_speakers:
            selected_speaker_df = input_df[input_df['speaker_id'] == speaker]
            final_train_data = final_train_data.append(selected_speaker_df)

        # sort based on speaker id
        final_train_data = final_train_data.sort_values(['relative_path'])
        final_test_data = final_test_data.sort_values(['relative_path'])

        return final_train_data, final_test_data



    def csv_size_reducer_random(self, input_df, max_speakers):

        # initiating the df
        final_data = pd.DataFrame(columns=['relative_path', 'speaker_id', 'test_type', 'session', 'file_length', 'user_id', 'mic_room', 'age',
                                           'WA', 'WR', 'intelligibility', 'diagnosis', 'surgery', 'gender', 'father_tongue', 'mother_tongue', 'chemo_therapy', 'dental_prosthesis', 'irradiation'])

        PEAKS_speaker_list = input_df['speaker_id'].unique().tolist()
        random.shuffle(PEAKS_speaker_list)

        # take "max_speakers" of speakers
        speakers = PEAKS_speaker_list[:max_speakers]

        # adding files based on chosen speakers
        for speaker in speakers:
            selected_speaker_df = input_df[input_df['speaker_id'] == speaker]
            final_data = final_data.append(selected_speaker_df)

        # sort based on speaker id
        final_data = final_data.sort_values(['relative_path'])

        return final_data



    def csv_size_reducer_fixed_age_based(self, input_df, max_speakers, lowerbound, upperbound):

        # initiating the df
        final_data = pd.DataFrame(columns=['relative_path', 'speaker_id', 'test_type', 'session', 'file_length', 'user_id', 'mic_room', 'age',
                                           'WA', 'WR', 'intelligibility', 'diagnosis', 'surgery', 'gender', 'father_tongue', 'mother_tongue', 'chemo_therapy', 'dental_prosthesis', 'irradiation'])

        # to remove the ones which are not given (denoted by -1)
        input_df = input_df[input_df['age'] > 0]

        PEAKS_speaker_list = input_df['speaker_id'].unique().tolist()

        # age speaker df
        age_speaker_df = pd.DataFrame(columns=['speaker_id', 'age'])

        for speaker in PEAKS_speaker_list:
            tempp = pd.DataFrame([[speaker, input_df[input_df['speaker_id'] == speaker]['age'].mean()]],
                                 columns=['speaker_id', 'age'])
            age_speaker_df = age_speaker_df.append(tempp)

        age_speaker_df = age_speaker_df[age_speaker_df['age'] >= lowerbound]
        age_speaker_df = age_speaker_df[age_speaker_df['age'] <= upperbound]

        chosen_speaker_list = age_speaker_df['speaker_id'].unique().tolist()
        random.shuffle(chosen_speaker_list)

        # take "max_speakers" of speakers
        chosen_speaker_list = chosen_speaker_list[:max_speakers]

        # adding files based on chosen ages
        for speaker in chosen_speaker_list:
            selected_speaker_df = input_df[input_df['speaker_id'] == speaker]
            final_data = final_data.append(selected_speaker_df)

        # sort based on speaker id
        final_data = final_data.sort_values(['relative_path'])

        return final_data



    def csv_size_reducer_std_mean_age_based(self, input_df, max_speakers, target_mean, target_std):

        # initiating the df
        final_data = pd.DataFrame(columns=['relative_path', 'speaker_id', 'test_type', 'session', 'file_length', 'user_id', 'mic_room', 'age',
                                           'WA', 'WR', 'intelligibility', 'diagnosis', 'surgery', 'gender', 'father_tongue', 'mother_tongue', 'chemo_therapy', 'dental_prosthesis', 'irradiation'])

        # to remove the ones which are not given (denoted by -1) and also some are zero
        input_df = input_df[input_df['age'] > 0]

        PEAKS_speaker_list = input_df['speaker_id'].unique().tolist()
        print(len(PEAKS_speaker_list))
        age_speaker_df = pd.DataFrame(columns=['speaker_id', 'age'])

        for speaker in PEAKS_speaker_list:
            tempp = pd.DataFrame([[speaker, input_df[input_df['speaker_id'] == speaker]['age'].mean()]],
                                 columns=['speaker_id', 'age'])
            age_speaker_df = age_speaker_df.append(tempp)

        mean_diff = 1000
        std_diff = 1000
        print('Beginning of While loop for sampling')
        while (mean_diff > 0.75 or std_diff > 0.75):
            sampled_df = age_speaker_df.sample(max_speakers)
            sampled_mean = sampled_df['age'].mean()
            sampled_std = sampled_df['age'].std()
            mean_diff = abs(target_mean - sampled_mean)
            std_diff = abs(target_std - sampled_std)

        print('End of While loop for sampling')
        chosen_speaker_list = sampled_df['speaker_id'].unique().tolist()

        # adding files based on chosen ages
        for speaker in chosen_speaker_list:
            selected_speaker_df = input_df[input_df['speaker_id'] == speaker]
            final_data = final_data.append(selected_speaker_df)

        # sort based on speaker id
        final_data = final_data.sort_values(['relative_path'])

        return final_data



    def trim_long_silences(self, wav):
        """
        Ensures that segments without voice in the waveform remain no longer than a
        threshold determined by the VAD parameters in params.py.

        Parameters
        ----------
        wav: numpy array of floats
            the raw waveform as a numpy array of floats

        Returns
        -------
        trimmed_wav: numpy array of floats
            the same waveform with silences trimmed
            away (length <= original wav length)
        """

        # Compute the voice detection window size
        samples_per_window = (self.params['preprocessing']['vad_window_length'] * self.params['preprocessing']['sr']) // 1000

        # Trim the end of the audio to have a multiple of the window size
        wav = wav[:len(wav) - (len(wav) % samples_per_window)]

        # Convert the float waveform to 16-bit mono PCM
        pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * int16_max)).astype(np.int16))

        # Perform voice activation detection
        voice_flags = []
        vad = webrtcvad.Vad(mode=3)
        for window_start in range(0, len(wav), samples_per_window):
            window_end = window_start + samples_per_window
            voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                             sample_rate=self.params['preprocessing']['sr']))
        voice_flags = np.array(voice_flags)

        # Smooth the voice detection with a moving average
        def moving_average(array, width):
            array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
            ret = np.cumsum(array_padded, dtype=float)
            ret[width:] = ret[width:] - ret[:-width]
            return ret[width - 1:] / width

        audio_mask = moving_average(voice_flags, self.params['preprocessing']['vad_moving_average_width'])
        audio_mask = np.round(audio_mask).astype(np.bool)

        # Dilate the voiced regions
        audio_mask = binary_dilation(audio_mask, np.ones(self.params['preprocessing']['vad_max_silence_length'] + 1))
        audio_mask = np.repeat(audio_mask, samples_per_window)

        return wav[audio_mask == True]


    def normalize_volume(self, wav, target_dBFS, increase_only=False, decrease_only=False):
        if increase_only and decrease_only:
            raise ValueError("Both increase only and decrease only are set")
        rms = np.sqrt(np.mean((wav * int16_max) ** 2))
        wave_dBFS = 20 * np.log10(rms / int16_max)
        dBFS_change = target_dBFS - wave_dBFS
        if dBFS_change < 0 and increase_only or dBFS_change > 0 and decrease_only:
            return wav
        return wav * (10 ** (dBFS_change / 20))


    def tisv_preproc_train(self, input_df, output_df_path, exp_name):
        """GE2E-loss-based pre-processing of training utterances for text-independent speaker verification.
        References:
            https://github.com/JanhHyun/Speaker_Verification/
            https://github.com/HarryVolek/PyTorch_Speaker_Verification/
            https://github.com/resemble-ai/Resemblyzer/

        Parameters
        ----------
        """
        # lower bound of utterance length
        utter_min_len = (self.params['preprocessing']['tisv_frame'] * self.params['preprocessing']['hop'] +
                         self.params['preprocessing']['window']) * self.params['preprocessing']['sr']

        final_dataframe = pd.DataFrame(columns=['relative_path', 'speaker_id', 'test_type', 'session', 'file_length', 'user_id', 'mic_room', 'age',
                                                'WA', 'WR', 'intelligibility', 'diagnosis', 'surgery', 'gender', 'father_tongue', 'mother_tongue', 'chemo_therapy', 'dental_prosthesis', 'irradiation'])

        for index, row in tqdm(input_df.iterrows()):

            utter_path = os.path.join(self.params['file_path'], row['relative_path'])
            utter, sr = sf.read(utter_path)

            # pre-processing and voice activity detection (VAD) part 1
            utter = self.normalize_volume(utter, self.params['preprocessing']['audio_norm_target_dBFS'], increase_only=True)
            utter = self.trim_long_silences(utter)
            if utter.shape[0] < utter_min_len:
                continue

            # basically this does nothing if 60 is chosen, 60 is too high, so the whole wav will be selected.
            # This just makes an interval from beginning to the end.
            intervals = librosa.effects.split(utter, top_db=30)  # voice activity detection part 2

            for interval_index, interval in enumerate(intervals):
                if (interval[1] - interval[0]) > utter_min_len:  # If partial utterance is sufficiently long,
                    utter_part = utter[interval[0]:interval[1]]
                    S = librosa.core.stft(y=utter_part, n_fft=self.params['preprocessing']['nfft'],
                                          win_length=int(self.params['preprocessing']['window'] * self.params['preprocessing']['sr']),
                                          hop_length=int(self.params['preprocessing']['hop'] * self.params['preprocessing']['sr']))
                    S = np.abs(S) ** 2
                    mel_basis = librosa.filters.mel(sr=self.params['preprocessing']['sr'], n_fft=self.params['preprocessing']['nfft'],
                                                    n_mels=self.params['preprocessing']['nmels'])
                    SS = np.log10(np.dot(mel_basis, S) + 1e-6)  # log mel spectrogram of partial utterance
                    os.makedirs(os.path.join(self.params['file_path'], 'tisv_preprocess', exp_name,
                                             os.path.dirname(row['relative_path'])), exist_ok=True)

                    rel_path = os.path.join(self.params['file_path'], 'tisv_preprocess', exp_name, os.path.dirname(row['relative_path']),
                                         os.path.basename(row['relative_path']).replace('.wav', '_interval_' + str(interval_index) + '.npy'))
                    np.save(rel_path, SS)

                    # add to the new dataframe
                    tempp = pd.DataFrame([[os.path.join('tisv_preprocess', exp_name,
                                                        os.path.dirname(row['relative_path']),
                                                        os.path.basename(row['relative_path']).replace('.wav', '_interval_' + str(interval_index) + '.npy')),
                                           row['speaker_id'], row['test_type'], row['session'],
                                           utter_part.shape[0] / self.params['preprocessing']['sr'], row['user_id'], row['mic_room'], row['age'],
                                           row['WA'], row['WR'], row['intelligibility'], row['diagnosis'], row['surgery'], row['gender'],
                                           row['father_tongue'], row['mother_tongue'], row['chemo_therapy'], row['dental_prosthesis'], row['irradiation']]],
                                         columns=['relative_path', 'speaker_id', 'test_type', 'session', 'file_length', 'user_id', 'mic_room', 'age',
                                                  'WA', 'WR', 'intelligibility', 'diagnosis', 'surgery', 'gender', 'father_tongue', 'mother_tongue', 'chemo_therapy', 'dental_prosthesis', 'irradiation'])
                    final_dataframe = final_dataframe.append(tempp)

        # to check the criterion of 8 utters after VAD
        final_dataframe = self.csv_speaker_trimmer(final_dataframe)
         # sort based on speaker id
        final_data = final_dataframe.sort_values(['relative_path'])
        final_data.to_csv(output_df_path, sep=';', index=False)



    def tisv_preproc_test(self, input_df, output_df_path, exp_name):
        """
        GE2E-loss-based pre-processing of validation & test utterances for text-independent speaker verification.
        References:
            https://github.com/JanhHyun/Speaker_Verification/
            https://github.com/HarryVolek/PyTorch_Speaker_Verification/
            https://github.com/resemble-ai/Resemblyzer/

        Parameters
        ----------
        """

        # lower bound of utterance length
        utter_min_len = (self.params['preprocessing']['tisv_frame'] * self.params['preprocessing']['hop'] +
                         self.params['preprocessing']['window']) * self.params['preprocessing']['sr']

        final_dataframe = pd.DataFrame(columns=['relative_path', 'speaker_id', 'test_type', 'session', 'file_length', 'user_id', 'mic_room',
                                                'age', 'WA', 'WR', 'intelligibility', 'diagnosis', 'surgery', 'gender', 'father_tongue', 'mother_tongue', 'chemo_therapy', 'dental_prosthesis', 'irradiation'])

        for index, row in tqdm(input_df.iterrows()):

            utter_path = os.path.join(self.params['file_path'], row['relative_path'])
            utter, sr = sf.read(utter_path)

            # pre-processing and voice activity detection (VAD) part 1
            utter = self.normalize_volume(utter, self.params['preprocessing']['audio_norm_target_dBFS'], increase_only=True)
            utter = self.trim_long_silences(utter)
            if utter.shape[0] < utter_min_len:
                continue

            # basically this does nothing if 60 is chosen, 60 is too high, so the whole wav will be selected.
            # This just makes an interval from beginning to the end.
            intervals = librosa.effects.split(utter, top_db=30)  # voice activity detection part 2

            for interval_index, interval in enumerate(intervals):
                if (interval[1] - interval[0]) > utter_min_len:  # If partial utterance is sufficiently long,
                    utter_part = utter[interval[0]:interval[1]]

                    # concatenate all the partial utterances of each utterance
                    if interval_index == 0:
                        utter_whole = utter_part
                    else:
                        try:
                            utter_whole = np.hstack((utter_whole, utter_part))
                        except:
                            utter_whole = utter_part
            if 'utter_whole' in locals():

                S = librosa.core.stft(y=utter_whole, n_fft=self.params['preprocessing']['nfft'],
                                      win_length=int(self.params['preprocessing']['window'] * self.params['preprocessing']['sr']),
                                      hop_length=int(self.params['preprocessing']['hop'] * self.params['preprocessing']['sr']))
                S = np.abs(S) ** 2
                mel_basis = librosa.filters.mel(sr=self.params['preprocessing']['sr'], n_fft=self.params['preprocessing']['nfft'],
                                                n_mels=self.params['preprocessing']['nmels'])

                SS = np.log10(np.dot(mel_basis, S) + 1e-6)  # log mel spectrogram of partial utterance
                os.makedirs(os.path.join(self.params['file_path'], 'tisv_preprocess', exp_name,
                                         os.path.dirname(row['relative_path'])), exist_ok=True)

                rel_path = os.path.join(self.params['file_path'], 'tisv_preprocess', exp_name, os.path.dirname(row['relative_path']),
                                        os.path.basename(row['relative_path']).replace('.wav', '.npy'))
                np.save(rel_path, SS)

                # add to the new dataframe
                tempp = pd.DataFrame([[os.path.join('tisv_preprocess', exp_name,
                                                    os.path.dirname(row['relative_path']),
                                                        os.path.basename(row['relative_path']).replace('.wav', '.npy')),
                                       row['speaker_id'], row['test_type'], row['session'],
                                       utter_whole.shape[0] / self.params['preprocessing']['sr'], row['user_id'], row['mic_room'], row['age'],
                                       row['WA'], row['WR'], row['intelligibility'], row['diagnosis'], row['surgery'],
                                       row['gender'], row['father_tongue'], row['mother_tongue'], row['chemo_therapy'], row['dental_prosthesis'], row['irradiation']]],
                                     columns=['relative_path', 'speaker_id', 'test_type', 'session', 'file_length', 'user_id', 'mic_room', 'age',
                                              'WA', 'WR', 'intelligibility', 'diagnosis', 'surgery', 'gender', 'father_tongue', 'mother_tongue', 'chemo_therapy', 'dental_prosthesis', 'irradiation'])
                final_dataframe = final_dataframe.append(tempp)

        # to check the criterion of 8 utters after VAD
        final_dataframe = self.csv_speaker_trimmer(final_dataframe)
         # sort based on speaker id
        final_data = final_dataframe.sort_values(['relative_path'])
        final_data.to_csv(output_df_path, sep=';', index=False)



    def csv_speaker_trimmer(self, input_df):
        """only keeps the speakers which have at least 8 utterances
        Parameters
        ----------
        Returns
        ----------
        """
        final_data = pd.DataFrame(columns=['relative_path', 'speaker_id', 'test_type', 'session', 'file_length', 'user_id', 'mic_room', 'age',
                                              'WA', 'WR', 'intelligibility', 'diagnosis', 'surgery', 'gender', 'father_tongue', 'mother_tongue', 'chemo_therapy', 'dental_prosthesis', 'irradiation'])

        list_speakers = input_df['speaker_id'].unique().tolist()

        for speaker in list_speakers:
            selected_speaker_df = input_df[input_df['speaker_id'] == speaker]
            if len(selected_speaker_df) >= 8:
                final_data = final_data.append(selected_speaker_df)

        # sort based on speaker id
        final_data = final_data.sort_values(['relative_path'])
        return final_data
