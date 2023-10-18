"""
speaker_main.py
Created on Nov 22, 2021.
Main file for training and testing for text independent speaker verification.

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de>
https://github.com/tayebiarasteh/
"""
import pdb
import os

import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from config.serde import open_experiment, create_experiment, delete_experiment
from data.speaker_data_loader import tisv_dataset_train_valid, tisv_dvector_creator_loader
from data.PEAKS_specific_data_preprocess import data_preprocess_PEAKS
from speaker_Train_Valid import Training
from speaker_Prediction import Prediction
from models.lstm import SpeechEmbedder
from models.speaker_loss import GE2ELoss

import warnings
warnings.filterwarnings('ignore')



def main_train(global_config_path="/PATH/config/config.yaml",
               valid=True, resume=False, experiment_name='name'):
    """Main function for training + validation of tisv based on GE2E.

    Parameters
    ----------
    global_config_path: str
        always global_config_path="/PATH/config/config.yaml"

    valid: bool
        if we want to do validation

    resume: bool
        if we are resuming training on a model

    experiment_name: str
        name of the experiment, in case of resuming training.
        name of new experiment, in case of new training.
    """

    if resume == True:
        params = open_experiment(experiment_name, global_config_path)
    else:
        params = create_experiment(experiment_name, global_config_path)
    cfg_path = params["cfg_path"]

    # Changeable network parameters
    loss_function = GE2ELoss
    optimizer = optim.Adam
    optimiser_params = {'lr': float(params['Network']['lr'])}

    model = SpeechEmbedder(nmels=params['preprocessing']['nmels'], hidden_dim=params['Network']['hidden_dim'],
                           output_dim=params['Network']['output_dim'], num_layers=params['Network']['num_layers'])

    trainer = Training(cfg_path, num_epochs=params['num_epochs'], resume=resume)
    if resume == True:
        trainer.load_checkpoint(model=model, optimiser=optimizer,
                        optimiser_params=optimiser_params, loss_function=loss_function)
    else:
        trainer.setup_model(model=model, optimiser=optimizer,
                        optimiser_params=optimiser_params, loss_function=loss_function)

    # loading the data
    train_dataset = tisv_dataset_train_valid(cfg_path=cfg_path, training=True, experiment_name=experiment_name)
    train_loader = DataLoader(train_dataset, batch_size=params['Network']['N'],
                              shuffle=True, num_workers=4, drop_last=True)
    if valid:
        valid_dataset = tisv_dataset_train_valid(cfg_path=cfg_path, training=False, experiment_name=experiment_name)
        valid_loader = DataLoader(valid_dataset, batch_size=params['Network']['N_valid'],
                                  shuffle=False, num_workers=4, drop_last=True)
    else:
        valid_loader = None

    trainer.execute_training(train_loader=train_loader, valid_loader=valid_loader, validiation=valid)



def main_dvector(global_config_path="/PATH/config/config.yaml",
                 experiment_name='GE2E_speaker'):
    """Main function for creating d-vectors of test and evaluation data
    and storing them to the memory.

    Parameters
    ----------
    global_config_path: str
        always global_config_path="/PATH/config/config.yaml"

    experiment_name: str
        the name of the experiment to be loaded
    """
    params = open_experiment(experiment_name, global_config_path)
    cfg_path = params['cfg_path']
    predictor = Prediction(cfg_path)
    model = SpeechEmbedder(nmels=params['preprocessing']['nmels'], hidden_dim=params['Network']['hidden_dim'],
                           output_dim=params['Network']['output_dim'], num_layers=params['Network']['num_layers'])
    predictor.setup_model(model=model)

    # d-vector creation
    data_handler = tisv_dvector_creator_loader(cfg_path=cfg_path, experiment_name=experiment_name)
    data_loader = data_handler.provide_data()
    predictor.dvector_prediction(data_loader)



def main_eval_test(global_config_path="/PATH/config/config.yaml",
                   experiment_name='GE2E_speaker', epochs=1000):
    """Main function for testing.

    Parameters
    ----------
    global_config_path: str
        always global_config_path="/PATH/config/config.yaml"

    experiment_name: str
        name of the experiment, in case of resuming training.
        name of new experiment, in case of new training.

    epochs: int
        total number of epochs to do the evaluation process.
        The results will be the average over the result of
        each epoch.
    """
    params = open_experiment(experiment_name, global_config_path)
    cfg_path = params['cfg_path']
    predictor = Prediction(cfg_path)

    # Threshold calculation
    threshold = predictor.thresholding(cfg_path, M=params['Network']['M_test'], epochs=epochs)
    # EER calculation
    predictor.predict(cfg_path, threshold=threshold, M=params['Network']['M_test'], epochs=epochs)



def main_dvector_eval_test_epochy(global_config_path="/PATH/config/config.yaml",
                   experiment_name='GE2E_speaker', epochs=1000):
    """Main function for creating d-vectors & testing, for different models based on epochs

    Parameters
    ----------
    global_config_path: str
        always global_config_path="/PATH/config/config.yaml"

    experiment_name: str
        name of the experiment, in case of resuming training.
        name of new experiment, in case of new training.

    epochs: int
        total number of epochs to do the evaluation process.
        The results will be the average over the result of
        each epoch.
    """
    params = open_experiment(experiment_name, global_config_path)
    cfg_path = params['cfg_path']
    predictor = Prediction(cfg_path)
    model = SpeechEmbedder(nmels=params['preprocessing']['nmels'], hidden_dim=params['Network']['hidden_dim'],
                           output_dim=params['Network']['output_dim'], num_layers=params['Network']['num_layers'])

    epoch_list = [400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200,
                  1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700, 1750, 1800, 1850, 1900]
    eer_M2 = np.zeros((len(epoch_list)))
    eer_M4 = np.zeros((len(epoch_list)))

    for idx, model_epoch in enumerate(epoch_list):
        predictor.setup_model(model=model, model_epoch=model_epoch)
        # d-vector creation
        data_handler = tisv_dvector_creator_loader(cfg_path=cfg_path, experiment_name=experiment_name)
        data_loader = data_handler.provide_data()
        predictor.dvector_prediction(data_loader)

        # Threshold calculation M = 2
        threshold_M2, avg_EER_eval_M2 = predictor.thresholding_epochy(cfg_path, M=4, epochs=epochs)
        # EER calculation M = 2
        avg_EER_test_M2, avg_FAR_test_M2, avg_FRR_test_M2 = predictor.predict_epochy(cfg_path, threshold=threshold_M2, M=4,
                                                                     epochs=epochs, model_epoch=model_epoch)

        # Threshold calculation M = 4
        threshold_M4, avg_EER_eval_M4 = predictor.thresholding_epochy(cfg_path, M=8, epochs=epochs)
        # EER calculation M = 4
        avg_EER_test_M4, avg_FAR_test_M4, avg_FRR_test_M4 = predictor.predict_epochy(cfg_path, threshold=threshold_M4, M=8,
                                                                     epochs=epochs, model_epoch=model_epoch)
        eer_M2[idx] = avg_EER_test_M2
        eer_M4[idx] = avg_EER_test_M4

        # for M = 2
        print('\n------------------------------------------------------'
              '----------------------------------')
        print(f'{experiment_name} | M: {2} \n')
        print(f'Model saved at epoch {model_epoch} | validation iterations: {epochs} ')
        print(f"\n\tAverage Test EER: {(avg_EER_test_M2) * 100:.2f}% | Fixed threshold: {threshold_M2:.2f} "
              f'\n\n\tAverage Evaluation EER: {(avg_EER_eval_M2) * 100:.2f}% | Average threshold: {threshold_M2:.2f}'
              f'\n\n\tTest FAR: {100 * avg_FAR_test_M2:.2f}% | '
              f'Test FRR: {100 * avg_FRR_test_M2:.2f}%\n')

        # saving the stats
        mesg = f'\n----------------------------------------------------------------------------------------\n' \
               f"{experiment_name} | Number of enrolment utterances (M): {2} \n" \
               f"Model saved at epoch {model_epoch} | validation iterations: {epochs} " \
               f"\n\n\tAverage Test EER: {(avg_EER_test_M2) * 100:.2f}% | Fixed threshold: {threshold_M2:.2f} " \
               f"\n\n\tAverage Evaluation EER: {(avg_EER_eval_M2) * 100:.2f}% | Average threshold: {threshold_M2:.2f}" \
               f'\n\n\tTest FAR: {100 * avg_FAR_test_M2:.2f}% | ' \
               f'Test FRR: {100 * avg_FRR_test_M2:.2f}%' \
               f'\n\n----------------------------------------------------------------------------------------\n'
        with open(os.path.join(params['target_dir'], params['stat_log_path']) + '/test_results_M2', 'a') as f:
            f.write(mesg)

        # for M = 4
        print('\n------------------------------------------------------'
              '----------------------------------')
        print(f'{experiment_name} | M: {4} \n')
        print(f'Model saved at epoch {model_epoch} | validation iterations: {epochs} ')
        print(f"\n\tAverage Test EER: {(avg_EER_test_M4) * 100:.2f}% | Fixed threshold: {threshold_M4:.2f} "
              f'\n\n\tAverage Evaluation EER: {(avg_EER_eval_M4) * 100:.2f}% | Average threshold: {threshold_M4:.2f}'
              f'\n\n\tTest FAR: {100 * avg_FAR_test_M4:.2f}% | '
              f'Test FRR: {100 * avg_FRR_test_M4:.2f}%\n')

        # saving the stats
        mesg = f'\n----------------------------------------------------------------------------------------\n' \
               f"{experiment_name} | Number of enrolment utterances (M): {4} \n" \
               f"Model saved at epoch {model_epoch} | validation iterations: {epochs} " \
               f"\n\n\tAverage Test EER: {(avg_EER_test_M4) * 100:.2f}% | Fixed threshold: {threshold_M4:.2f} " \
               f"\n\n\tAverage Evaluation EER: {(avg_EER_eval_M4) * 100:.2f}% | Average threshold: {threshold_M4:.2f}" \
               f'\n\n\tTest FAR: {100 * avg_FAR_test_M4:.2f}% | ' \
               f'Test FRR: {100 * avg_FRR_test_M4:.2f}%' \
               f'\n\n----------------------------------------------------------------------------------------\n'
        with open(os.path.join(params['target_dir'], params['stat_log_path']) + '/test_results_M4', 'a') as f:
            f.write(mesg)

    fig = plt.figure()
    plt.plot([400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200,
                  1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700, 1750, 1800, 1850, 1900], eer_M2*100, label='M2')
    plt.plot([400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200,
                  1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700, 1750, 1800, 1850, 1900], eer_M4*100, label='M4')
    plt.grid()
    plt.legend(loc="upper right")
    plt.xlabel('Epoch')
    plt.ylabel('% EER')
    plt.title(experiment_name)
    fig.savefig(os.path.join(params['target_dir'], params['stat_log_path'], 'eer.png'))





def main_train_test_scatterplot(global_config_path="/PATH/config/config.yaml",
               valid=False, resume=False, experiment_name='name', epochs=500):
    """

    Parameters
    ----------
    global_config_path: str
        always global_config_path="/PATH/config/config.yaml"

    valid: bool
        if we want to do validation

    resume: bool
        if we are resuming training on a model

    experiment_name: str
        name of the experiment, in case of resuming training.
        name of new experiment, in case of new training.
    """

    if resume == True:
        params = open_experiment(experiment_name, global_config_path)
    else:
        params = create_experiment(experiment_name, global_config_path)
    cfg_path = params["cfg_path"]

    # Changeable network parameters
    loss_function = GE2ELoss
    optimizer = optim.Adam
    optimiser_params = {'lr': float(params['Network']['lr'])}

    model = SpeechEmbedder(nmels=params['preprocessing']['nmels'], hidden_dim=params['Network']['hidden_dim'],
                           output_dim=params['Network']['output_dim'], num_layers=params['Network']['num_layers'])

    trainer = Training(cfg_path, num_epochs=params['num_epochs'], resume=resume)
    if resume == True:
        trainer.load_checkpoint(model=model, optimiser=optimizer,
                                optimiser_params=optimiser_params, loss_function=loss_function)
    else:
        trainer.setup_model(model=model, optimiser=optimizer,
                            optimiser_params=optimiser_params, loss_function=loss_function)

    # loading the data
    train_dataset = tisv_dataset_train_valid(cfg_path=cfg_path, training=True, experiment_name=experiment_name)
    train_loader = DataLoader(train_dataset, batch_size=params['Network']['N'],
                              shuffle=True, num_workers=4, drop_last=True)
    if valid:
        valid_dataset = tisv_dataset_train_valid(cfg_path=cfg_path, training=False, experiment_name=experiment_name)
        valid_loader = DataLoader(valid_dataset, batch_size=params['Network']['N_valid'],
                                  shuffle=False, num_workers=4, drop_last=True)
    else:
        valid_loader = None

    trainer.execute_training(train_loader=train_loader, valid_loader=valid_loader, validiation=valid)

    ####################### testing ####################

    params = open_experiment(experiment_name, global_config_path)
    cfg_path = params['cfg_path']
    predictor = Prediction(cfg_path)
    model = SpeechEmbedder(nmels=params['preprocessing']['nmels'], hidden_dim=params['Network']['hidden_dim'],
                           output_dim=params['Network']['output_dim'], num_layers=params['Network']['num_layers'])

    epoch_list = [1, 2]
    eer_M2 = np.zeros((len(epoch_list)))
    eer_M4 = np.zeros((len(epoch_list)))

    for idx, model_epoch in enumerate(epoch_list):
        predictor.setup_model(model=model, model_epoch=model_epoch)
        # d-vector creation
        data_handler = tisv_dvector_creator_loader(cfg_path=cfg_path, experiment_name=experiment_name)
        data_loader = data_handler.provide_data()
        predictor.dvector_prediction(data_loader)

        # Threshold calculation M = 2
        threshold_M2, avg_EER_eval_M2 = predictor.thresholding_epochy(cfg_path, M=4, epochs=epochs)
        # EER calculation M = 2
        avg_EER_test_M2, avg_FAR_test_M2, avg_FRR_test_M2 = predictor.predict_epochy(cfg_path,
                                                                                     threshold=threshold_M2, M=4,
                                                                                     epochs=epochs,
                                                                                     model_epoch=model_epoch)

        # Threshold calculation M = 4
        threshold_M4, avg_EER_eval_M4 = predictor.thresholding_epochy(cfg_path, M=8, epochs=epochs)
        # EER calculation M = 4
        avg_EER_test_M4, avg_FAR_test_M4, avg_FRR_test_M4 = predictor.predict_epochy(cfg_path,
                                                                                     threshold=threshold_M4, M=8,
                                                                                     epochs=epochs,
                                                                                     model_epoch=model_epoch)
        eer_M2[idx] = avg_EER_test_M2
        eer_M4[idx] = avg_EER_test_M4

        # for M = 2
        print('\n------------------------------------------------------'
              '----------------------------------')
        print(f'{experiment_name} | M: {2} \n')
        print(f'Model saved at epoch {model_epoch} | validation iterations: {epochs} ')
        print(f"\n\tAverage Test EER: {(avg_EER_test_M2) * 100:.2f}% | Fixed threshold: {threshold_M2:.2f} "
              f'\n\n\tAverage Evaluation EER: {(avg_EER_eval_M2) * 100:.2f}% | Average threshold: {threshold_M2:.2f}'
              f'\n\n\tTest FAR: {100 * avg_FAR_test_M2:.2f}% | '
              f'Test FRR: {100 * avg_FRR_test_M2:.2f}%\n')

        # saving the stats
        mesg = f'\n----------------------------------------------------------------------------------------\n' \
               f"{experiment_name} | Number of enrolment utterances (M): {2} \n" \
               f"Model saved at epoch {model_epoch} | validation iterations: {epochs} " \
               f"\n\n\tAverage Test EER: {(avg_EER_test_M2) * 100:.2f}% | Fixed threshold: {threshold_M2:.2f} " \
               f"\n\n\tAverage Evaluation EER: {(avg_EER_eval_M2) * 100:.2f}% | Average threshold: {threshold_M2:.2f}" \
               f'\n\n\tTest FAR: {100 * avg_FAR_test_M2:.2f}% | ' \
               f'Test FRR: {100 * avg_FRR_test_M2:.2f}%' \
               f'\n\n----------------------------------------------------------------------------------------\n'
        with open(os.path.join(params['target_dir'], params['stat_log_path']) + '/test_results_M2', 'a') as f:
            f.write(mesg)

        # for M = 4
        print('\n------------------------------------------------------'
              '----------------------------------')
        print(f'{experiment_name} | M: {4} \n')
        print(f'Model saved at epoch {model_epoch} | validation iterations: {epochs} ')
        print(f"\n\tAverage Test EER: {(avg_EER_test_M4) * 100:.2f}% | Fixed threshold: {threshold_M4:.2f} "
              f'\n\n\tAverage Evaluation EER: {(avg_EER_eval_M4) * 100:.2f}% | Average threshold: {threshold_M4:.2f}'
              f'\n\n\tTest FAR: {100 * avg_FAR_test_M4:.2f}% | '
              f'Test FRR: {100 * avg_FRR_test_M4:.2f}%\n')

        # saving the stats
        mesg = f'\n----------------------------------------------------------------------------------------\n' \
               f"{experiment_name} | Number of enrolment utterances (M): {4} \n" \
               f"Model saved at epoch {model_epoch} | validation iterations: {epochs} " \
               f"\n\n\tAverage Test EER: {(avg_EER_test_M4) * 100:.2f}% | Fixed threshold: {threshold_M4:.2f} " \
               f"\n\n\tAverage Evaluation EER: {(avg_EER_eval_M4) * 100:.2f}% | Average threshold: {threshold_M4:.2f}" \
               f'\n\n\tTest FAR: {100 * avg_FAR_test_M4:.2f}% | ' \
               f'Test FRR: {100 * avg_FRR_test_M4:.2f}%' \
               f'\n\n----------------------------------------------------------------------------------------\n'
        with open(os.path.join(params['target_dir'], params['stat_log_path']) + '/test_results_M4', 'a') as f:
            f.write(mesg)

    fig = plt.figure()
    plt.plot([1, 2], eer_M2 * 100, label='M2')
    plt.plot([1, 2], eer_M4 * 100, label='M4')
    plt.grid()
    plt.legend(loc="upper right")
    plt.xlabel('Epoch')
    plt.ylabel('% EER')
    plt.title(experiment_name)
    fig.savefig(os.path.join(params['target_dir'], params['stat_log_path'], 'eer.png'))

    eer_together = eer_M2 + eer_M4

    min_index = np.argmin(eer_together)

    epoch_num = epoch_list[min_index]

    print('best epoch:', epoch_num)

    mesg = f'\n----------------------------------------------------------------------------------------\n' \
    f'\n----------------------------------------------------------------------------------------\n' \
    f'\n----------------------------------------------------------------------------------------\n' \
           f"\nbest epoch: {epoch_num}" \
           f'\n\n----------------------------------------------------------------------------------------\n'
    with open(os.path.join(params['target_dir'], params['stat_log_path']) + '/test_results_M4', 'a') as f:
        f.write(mesg)

    #########################################################################
    #########################################################################


    params = open_experiment(experiment_name, global_config_path)
    cfg_path = params['cfg_path']

    predictor = Prediction(cfg_path)
    model = SpeechEmbedder(nmels=params['preprocessing']['nmels'], hidden_dim=params['Network']['hidden_dim'],
                           output_dim=params['Network']['output_dim'], num_layers=params['Network']['num_layers'])
    predictor.setup_model(model=model, model_epoch=epoch_num)

    # d-vector creation
    data_handler = tisv_dvector_creator_loader(cfg_path=cfg_path, experiment_name=experiment_name)
    data_loader = data_handler.provide_data()
    predictor.dvector_prediction(data_loader)

    # Threshold calculation
    threshold = predictor.thresholding(cfg_path, M=4, epochs=epochs)
    # EER calculation
    test_results_csv_M2 = predictor.predict_forscatter(cfg_path, threshold=threshold, M=4, epochs=epochs, model_epoch=epoch_num, experiment_name=experiment_name, speaker_num=len(data_loader))

    # Threshold calculation
    threshold = predictor.thresholding(cfg_path, M=8, epochs=epochs)
    # EER calculation
    test_results_csv_M4 = predictor.predict_forscatter(cfg_path, threshold=threshold, M=8, epochs=epochs, model_epoch=epoch_num, experiment_name=experiment_name, speaker_num=len(data_loader))
    test_results_csv = test_results_csv_M2.append(test_results_csv_M4)
    test_results_csv.to_csv(os.path.join(params['target_dir'], params['stat_log_path']) + '/test_results.csv', sep=';', index=False)



def main_eval_test_forscattering(global_config_path="/PATH/config/config.yaml",
                   experiment_name='GE2E_speaker', epochs=1000):
    """

    Parameters
    ----------
    global_config_path: str
        always global_config_path="/PATH/config/config.yaml"

    experiment_name: str
        name of the experiment, in case of resuming training.
        name of new experiment, in case of new training.

    epochs: int
        total number of epochs to do the evaluation process.
        The results will be the average over the result of
        each epoch.
    """
    params = open_experiment(experiment_name, global_config_path)
    cfg_path = params['cfg_path']
    predictor = Prediction(cfg_path)
    model = SpeechEmbedder(nmels=params['preprocessing']['nmels'], hidden_dim=params['Network']['hidden_dim'],
                           output_dim=params['Network']['output_dim'], num_layers=params['Network']['num_layers'])
    predictor.setup_model(model=model, model_epoch=2)

    # d-vector creation
    data_handler = tisv_dvector_creator_loader(cfg_path=cfg_path, experiment_name=experiment_name)
    data_loader = data_handler.provide_data()
    predictor.dvector_prediction(data_loader)

    # Threshold calculation
    threshold = predictor.thresholding(cfg_path, M=params['Network']['M_test'], epochs=epochs)

    # EER calculation
    test_results_csv_M2 = predictor.predict_forscatter(cfg_path, threshold=threshold, M=4, epochs=epochs, model_epoch=2, experiment_name=experiment_name, speaker_num=len(data_loader))
    test_results_csv_M4 = predictor.predict_forscatter(cfg_path, threshold=threshold, M=8, epochs=epochs, model_epoch=2, experiment_name=experiment_name, speaker_num=len(data_loader))
    test_results_csv = test_results_csv_M2.append(test_results_csv_M4)
    test_results_csv.to_csv(os.path.join(params['target_dir'], params['stat_log_path']) + '/test_results.csv', sep=';', index=False)




