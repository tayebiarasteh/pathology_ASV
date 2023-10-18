"""
speaker_prediction.py
Created on Nov 22, 2021.
Prediction (test) class = evaluation + testing

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de>
https://github.com/tayebiarasteh/
"""

import time
import random
import pdb
from tqdm import tqdm
import os
import numpy as np
from matplotlib import pyplot as plt
import torch
import pandas as pd

from data.speaker_data_loader import tisv_after_dvector_loader, tisv_after_dvector_loader_forscattering
from config.serde import read_config
from utils.utils import get_centroids, get_cossim


class Prediction:
    def __init__(self, cfg_path):
        self.params = read_config(cfg_path)
        self.cfg_path = cfg_path
        self.setup_cuda()


    def setup_cuda(self, cuda_device_id=0):
        """setup the device.

        Parameters
        ----------
        cuda_device_id: int
            cuda device id
        """
        if torch.cuda.is_available():
            torch.backends.cudnn.fastest = True
            torch.cuda.set_device(cuda_device_id)
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')


    def time_duration(self, start_time, end_time):
        """calculating the duration of training or one iteration

        Parameters
        ----------
        start_time: float
            starting time of the operation

        end_time: float
            ending time of the operation

        Returns
        -------
        elapsed_hours: int
            total hours part of the elapsed time

        elapsed_mins: int
            total minutes part of the elapsed time

        elapsed_secs: int
            total seconds part of the elapsed time
        """
        elapsed_time = end_time - start_time
        elapsed_hours = int(elapsed_time / 3600)
        if elapsed_hours >= 1:
            elapsed_mins = int((elapsed_time / 60) - (elapsed_hours * 60))
            elapsed_secs = int(elapsed_time - (elapsed_hours * 3600) - (elapsed_mins * 60))
        else:
            elapsed_mins = int(elapsed_time / 60)
            elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_hours, elapsed_mins, elapsed_secs


    def setup_model(self, model, model_file_name=None, model_epoch=400):
        if model_file_name == None:
            model_file_name = self.params['trained_model_name']
        self.model = model.to(self.device)
        self.model.load_state_dict(torch.load(os.path.join(self.params['target_dir'], self.params['network_output_path'], "epoch" + str(model_epoch) +"_" + model_file_name)))



    def dvector_prediction(self, test_loader):
        """
        Prediction
        For d-vector creation (prediction of the input utterances)
        """
        self.params = read_config(self.cfg_path)
        self.model.eval()

        with torch.no_grad():
            # loop over speakers
            for speaker_name in tqdm(test_loader):
                embeddings_list = []
                speaker = test_loader[speaker_name]
                # loop over utterances
                for utterance in speaker:

                    features = []
                    # sliding window
                    for i in range(utterance.shape[0]//80):
                        if i == (utterance.shape[0]//80) - 1:
                            features.append(utterance[-160:])
                        else:
                            features.append(utterance[i * 80: i * 80 + 160])
                    features = torch.stack(features)
                    features = features.to(self.device)

                    dvector = self.model(features)
                    dvector = torch.mean(dvector, dim=0)
                    dvector = dvector.cpu().numpy()
                    embeddings_list.append(dvector)

                embeddings = np.array(embeddings_list)
                # save embedding as numpy file
                np.save(os.path.join(os.path.join(self.params['target_dir'], self.params['dvectors_path']), str(speaker_name) + ".npy"), embeddings)



    def thresholding(self, cfg_path, M=14, epochs=10):
        """
        evaluation (enrolment + verification)
        Open-set
        :epochs: because we are sampling each time, we have something like epoch here in testing
        """
        total_start_time = time.time()
        avg_EER = 0
        avg_FAR = 0
        avg_FRR = 0
        avg_thresh = 0
        FAR_plot = 0
        FRR_plot = 0

        for _ in tqdm(range(epochs)):
            dvector_dataset = tisv_after_dvector_loader(cfg_path=cfg_path, M=M)
            dvector_loader = dvector_dataset.provide_test()
            assert M % 2 == 0
            enrollment_embeddings, verification_embeddings = torch.split(dvector_loader, int(dvector_loader.size(1) // 2), dim=1)

            enrollment_centroids = get_centroids(enrollment_embeddings)
            sim_matrix = get_cossim(verification_embeddings, enrollment_centroids)

            # calculating EER
            diff = 1
            EER = 0
            EER_thresh = 0
            EER_FAR = 0
            EER_FRR = 0
            FAR_temp = []
            FRR_temp = []
            thres_temp = []

            for thres in [0.01 * i + 0.20 for i in range(60)]:
                sim_matrix_thresh = sim_matrix > thres
                FAR = (sum([sim_matrix_thresh[i].float().sum() - sim_matrix_thresh[i, :, i].float().sum() for i in
                            range(int(dvector_loader.shape[0]))]) / (dvector_loader.shape[0] - 1.0) / (float(M / 2)) / dvector_loader.shape[0])
                FRR = (sum([M / 2 - sim_matrix_thresh[i, :, i].float().sum() for i in
                            range(int(dvector_loader.shape[0]))]) / (float(M / 2)) / dvector_loader.shape[0])

                FAR_temp.append(FAR*100)
                FRR_temp.append(FRR*100)
                thres_temp.append(thres)

                # Save threshold when FAR = FRR (=EER)
                if diff > abs(FAR - FRR):
                    diff = abs(FAR - FRR)
                    EER = (FAR + FRR) / 2
                    EER_thresh = thres
                    EER_FAR = FAR
                    EER_FRR = FRR

            avg_EER += EER
            avg_FAR += EER_FAR
            avg_FRR += EER_FRR
            avg_thresh += EER_thresh
            FAR_plot += np.asarray(FAR_temp)
            FRR_plot += np.asarray(FRR_temp)
            del dvector_dataset
            del dvector_loader

        end_time = time.time()
        total_hours, total_mins, total_secs = self.time_duration(total_start_time, end_time)
        print('\n------------------------------------------------------'
              '----------------------------------')
        print(f'Total Time across validation {epochs} iterations: {total_hours}h {total_mins}m {total_secs}s')
        print(f"\n\tAverage Eval EER: {(avg_EER / epochs) * 100:.2f}% | "
              f'\n\tThreshold: {avg_thresh / epochs:.2f} | '
              f'Eval FAR: {100 * avg_FAR / epochs:.2f}% | '
              f'Eval FRR: {100 * avg_FRR / epochs:.2f}%')

        return avg_thresh / epochs



    def predict(self, cfg_path, threshold=0.5, M=14, epochs=10, model_epoch=400):
        """
        Testing (enrolment + verification)
        Open-set
        :epochs: because we are sampling each time, we have something like epoch here in testing
        """
        total_start_time = time.time()
        avg_EER = 0
        avg_FAR = 0
        avg_FRR = 0

        for _ in tqdm(range(epochs)):
            dvector_dataset = tisv_after_dvector_loader(cfg_path=cfg_path, M=M)
            dvector_loader = dvector_dataset.provide_test()

            assert M % 2 == 0
            enrollment_embeddings, verification_embeddings = torch.split(dvector_loader, int(dvector_loader.size(1) // 2), dim=1)

            enrollment_centroids = get_centroids(enrollment_embeddings)
            sim_matrix = get_cossim(verification_embeddings, enrollment_centroids)

            # calculating EER
            sim_matrix_thresh = sim_matrix > threshold
            FAR = (sum([sim_matrix_thresh[i].float().sum() - sim_matrix_thresh[i, :, i].float().sum() for i in
                        range(int(dvector_loader.shape[0]))]) / (dvector_loader.shape[0] - 1.0) / (float(M / 2)) /
                   dvector_loader.shape[0])
            FRR = (sum([M / 2 - sim_matrix_thresh[i, :, i].float().sum() for i in
                        range(int(dvector_loader.shape[0]))]) / (float(M / 2)) / dvector_loader.shape[0])

            # Save threshold when FAR = FRR (=EER)
            EER = (FAR + FRR) / 2
            avg_EER += EER
            avg_FAR += FAR
            avg_FRR += FRR

            del dvector_dataset
            del dvector_loader

        end_time = time.time()
        total_hours, total_mins, total_secs = self.time_duration(total_start_time, end_time)
        print('\n------------------------------------------------------'
              '----------------------------------')
        print(f'Total Time across {epochs} validation iterations: {total_hours}h {total_mins}m {total_secs}s')
        print(f"\n\tAverage Test EER: {(avg_EER / epochs) * 100:.2f}% | "
              f'\n\n\tTest FAR: {100 * avg_FAR / epochs:.2f}% | '
              f'Test FRR: {100 * avg_FRR / epochs:.2f}%')

        # saving the stats
        mesg = f'\n----------------------------------------------------------------------------------------\n' \
               f"Model saved at epoch {model_epoch} | {epochs} validation iterations. " \
               f"\n\n\tAverage Test EER: {(avg_EER / epochs) * 100:.2f}% | " \
               f'\n\n\tTest FAR: {100 * avg_FAR / epochs:.2f}% | ' \
               f'Test FRR: {100 * avg_FRR / epochs:.2f}%' \
               f'\n\n----------------------------------------------------------------------------------------\n'
        with open(os.path.join(self.params['target_dir'], self.params['stat_log_path']) + '/test_results', 'a') as f:
            f.write(mesg)



    def predict_forscatter(self, cfg_path, threshold=0.5, M=14, epochs=10, model_epoch=400, experiment_name='name', speaker_num=19):
        """
        Testing (enrolment + verification)
        Open-set
        :epochs: because we are sampling each time, we have something like epoch here in testing
        """
        total_start_time = time.time()
        avg_EER = 0
        avg_FAR = 0
        avg_FRR = 0
        EER_list = np.zeros(speaker_num)
        WR_list = np.zeros(speaker_num)

        for _ in tqdm(range(epochs)):
            dvector_dataset = tisv_after_dvector_loader_forscattering(cfg_path=cfg_path, M=M, experiment_name=experiment_name)
            dvector_loader, output_WR_list, speaker_name_list, diagnosis_list, age_list, output_WA_list, intelligibility_list, user_id_list, mic_room_list, gender_list = dvector_dataset.provide_test()

            assert M % 2 == 0
            enrollment_embeddings, verification_embeddings = torch.split(dvector_loader, int(dvector_loader.size(1) // 2), dim=1)

            enrollment_centroids = get_centroids(enrollment_embeddings)
            sim_matrix = get_cossim(verification_embeddings, enrollment_centroids)

            ########################################################################################################
            # calculating EER
            sim_matrix_thresh = sim_matrix > threshold

            FAR_list = []
            FRR_list = []

            FAR = (sum([sim_matrix_thresh[i].float().sum() - sim_matrix_thresh[i, :, i].float().sum() for i in
                        range(int(dvector_loader.shape[0]))]) / (dvector_loader.shape[0] - 1.0) / (float(M / 2)) /
                   dvector_loader.shape[0])

            for i in range(int(dvector_loader.shape[0])):
                FAR_list.append((sim_matrix_thresh[i].float().sum() - sim_matrix_thresh[i, :, i].float().sum()) / (
                            dvector_loader.shape[0] - 1.0) / (float(M / 2)))

            FRR = (sum([M / 2 - sim_matrix_thresh[i, :, i].float().sum() for i in
                        range(int(dvector_loader.shape[0]))]) / (float(M / 2)) / dvector_loader.shape[0])

            for i in range(int(dvector_loader.shape[0])):
                FRR_list.append(((M / 2 - sim_matrix_thresh[i, :, i].float().sum()) / (float(M / 2))))

            FAR_list = np.stack(FAR_list, 0)
            FRR_list = np.stack(FRR_list, 0)
            ################################################################

            # Save threshold when FAR = FRR (=EER)
            EER = (FAR + FRR) / 2
            avg_EER += EER
            avg_FAR += FAR
            avg_FRR += FRR
            EER_list += ((FAR_list + FRR_list) / 2)
            WR_list += output_WR_list

            del dvector_dataset
            del dvector_loader

        final_EER = EER_list / epochs
        final_EER *= 100
        final_WR = WR_list / epochs

        end_time = time.time()
        total_hours, total_mins, total_secs = self.time_duration(total_start_time, end_time)
        print('\n------------------------------------------------------'
              '----------------------------------')
        print(f'Total Time across {epochs} validation iterations: {total_hours}h {total_mins}m {total_secs}s')
        print(f"\n\tAverage Test EER: {(avg_EER / epochs) * 100:.2f}% | "
              f'\n\n\tTest FAR: {100 * avg_FAR / epochs:.2f}% | '
              f'Test FRR: {100 * avg_FRR / epochs:.2f}%')

        # saving the stats
        mesg = f'\n----------------------------------------------------------------------------------------\n' \
               f"Model saved at epoch {model_epoch} | {epochs} validation iterations. " \
               f"\n\n\tAverage Test EER: {(avg_EER / epochs) * 100:.2f}% | " \
               f'\n\n\tTest FAR: {100 * avg_FAR / epochs:.2f}% | ' \
               f'Test FRR: {100 * avg_FRR / epochs:.2f}%' \
               f'\n\n----------------------------------------------------------------------------------------\n'
        with open(os.path.join(self.params['target_dir'], self.params['stat_log_path']) + '/test_results', 'a') as f:
            f.write(mesg)


        output_df = pd.DataFrame({'speaker_id': speaker_name_list, 'user_id': user_id_list, 'mic_room': mic_room_list, 'EER': final_EER, 'WR': final_WR, 'diagnosis': diagnosis_list,
                                  'age': age_list, 'WA': output_WA_list, 'intelligibility': intelligibility_list, 'gender': gender_list})
        output_df = output_df.round({"EER": 2, "WR": 2, "age": 2, "WA": 2})

        output_df.to_csv(os.path.join(self.params['target_dir'], self.params['stat_log_path']) + '/WR_EER_scatter_plot_M' + str(int(M / 2)) + '.csv', sep=';', index=False)

        output_df = output_df[output_df['WR'] > - 100]

        correl = np.corrcoef(output_df['EER'].values, output_df['WR'].values)[1,0]
        test_results_csv = pd.DataFrame([['M' + str(int(M / 2)), (avg_EER.item() / epochs) * 100, correl, model_epoch]], columns=['M', 'EER', 'CORREL', 'epoch_num'])
        test_results_csv = test_results_csv.round({"EER": 2, "CORREL": 4})

        fig = plt.figure()
        plt.scatter(output_df['WR'], output_df['EER'])
        plt.xlabel('WR [%]')
        plt.ylabel('EER [%]')
        plt.title(experiment_name + '_M=' + str(int(M / 2)))
        plt.grid()
        # plt.show()
        fig.savefig(os.path.join(self.params['target_dir'], self.params['stat_log_path'], 'WR_EER_scatter_plot_M' + str(int(M / 2)) + '.png'))

        return test_results_csv



    def thresholding_epochy(self, cfg_path, M=14, epochs=10):
        """
        evaluation (enrolment + verification)
        Open-set
        :epochs: because we are sampling each time, we have something like epoch here in testing
        """
        avg_EER = 0
        avg_FAR = 0
        avg_FRR = 0
        avg_thresh = 0
        FAR_plot = 0
        FRR_plot = 0

        for _ in tqdm(range(epochs)):
            dvector_dataset = tisv_after_dvector_loader(cfg_path=cfg_path, M=M)
            dvector_loader = dvector_dataset.provide_test()
            assert M % 2 == 0
            enrollment_embeddings, verification_embeddings = torch.split(dvector_loader, int(dvector_loader.size(1) // 2), dim=1)

            enrollment_centroids = get_centroids(enrollment_embeddings)
            sim_matrix = get_cossim(verification_embeddings, enrollment_centroids)

            # calculating EER
            diff = 1
            EER = 0
            EER_thresh = 0
            EER_FAR = 0
            EER_FRR = 0
            FAR_temp = []
            FRR_temp = []
            thres_temp = []

            for thres in [0.01 * i + 0.20 for i in range(60)]:
                sim_matrix_thresh = sim_matrix > thres
                FAR = (sum([sim_matrix_thresh[i].float().sum() - sim_matrix_thresh[i, :, i].float().sum() for i in
                            range(int(dvector_loader.shape[0]))]) / (dvector_loader.shape[0] - 1.0) / (float(M / 2)) / dvector_loader.shape[0])
                FRR = (sum([M / 2 - sim_matrix_thresh[i, :, i].float().sum() for i in
                            range(int(dvector_loader.shape[0]))]) / (float(M / 2)) / dvector_loader.shape[0])

                FAR_temp.append(FAR*100)
                FRR_temp.append(FRR*100)
                thres_temp.append(thres)

                # Save threshold when FAR = FRR (=EER)
                if diff > abs(FAR - FRR):
                    diff = abs(FAR - FRR)
                    EER = (FAR + FRR) / 2
                    EER_thresh = thres
                    EER_FAR = FAR
                    EER_FRR = FRR

            avg_EER += EER
            avg_FAR += EER_FAR
            avg_FRR += EER_FRR
            avg_thresh += EER_thresh
            FAR_plot += np.asarray(FAR_temp)
            FRR_plot += np.asarray(FRR_temp)
            del dvector_dataset
            del dvector_loader

        return avg_thresh / epochs, avg_EER / epochs



    def predict_epochy(self, cfg_path, threshold=0.5, M=14, epochs=10, model_epoch=400):
        """
        Testing (enrolment + verification)
        Open-set
        :epochs: because we are sampling each time, we have something like epoch here in testing
        """
        avg_EER = 0
        avg_FAR = 0
        avg_FRR = 0

        for _ in tqdm(range(epochs)):
            dvector_dataset = tisv_after_dvector_loader(cfg_path=cfg_path, M=M)
            dvector_loader = dvector_dataset.provide_test()

            assert M % 2 == 0
            enrollment_embeddings, verification_embeddings = torch.split(dvector_loader, int(dvector_loader.size(1) // 2), dim=1)

            enrollment_centroids = get_centroids(enrollment_embeddings)
            sim_matrix = get_cossim(verification_embeddings, enrollment_centroids)

            # calculating EER
            sim_matrix_thresh = sim_matrix > threshold
            FAR = (sum([sim_matrix_thresh[i].float().sum() - sim_matrix_thresh[i, :, i].float().sum() for i in
                        range(int(dvector_loader.shape[0]))]) / (dvector_loader.shape[0] - 1.0) / (float(M / 2)) /
                   dvector_loader.shape[0])
            FRR = (sum([M / 2 - sim_matrix_thresh[i, :, i].float().sum() for i in
                        range(int(dvector_loader.shape[0]))]) / (float(M / 2)) / dvector_loader.shape[0])

            # Save threshold when FAR = FRR (=EER)
            EER = (FAR + FRR) / 2
            avg_EER += EER
            avg_FAR += FAR
            avg_FRR += FRR

            del dvector_dataset
            del dvector_loader

        return avg_EER / epochs, avg_FAR / epochs, avg_FRR / epochs