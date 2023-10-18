"""
speaker_Train_Valid.py
Created on Nov 22, 2021.
Training and Validation classes Text-independent Speaker verification, GE2E end to end

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de>
https://github.com/tayebiarasteh/
"""

import os.path
import time
import pdb
from tensorboardX import SummaryWriter
import torch
import random

from config.serde import read_config, write_config
from utils.utils import get_centroids, get_cossim

import warnings
warnings.filterwarnings('ignore')



class Training:
    def __init__(self, cfg_path, num_epochs=10, resume=False, torch_seed=None):
        """This class represents training and validation processes.

        Parameters
        ----------
        cfg_path: str
            Config file path of the experiment

        num_epochs: int
            Total number of iterations for training

        resume: bool
            if we are resuming training from a checkpoint

        torch_seed: int
            Seed used for random generators in PyTorch functions
        """
        self.params = read_config(cfg_path)
        self.cfg_path = cfg_path
        self.num_epochs = num_epochs

        if resume == False:
            self.model_info = self.params['Network']
            self.model_info['seed'] = torch_seed or self.model_info['seed']
            self.epoch = 0
            self.best_loss = float('inf')
            self.setup_cuda()
            self.writer = SummaryWriter(log_dir=os.path.join(self.params['target_dir'], self.params['tb_logs_path']))


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
            torch.cuda.manual_seed_all(self.model_info['seed'])
            torch.manual_seed(self.model_info['seed'])
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


    def setup_model(self, model, optimiser, optimiser_params, loss_function):
        """Setting up all the models, optimizers, and loss functions.

        Parameters
        ----------
        model: model file
            The network

        optimiser: optimizer file
            The optimizer

        optimiser_params: optimizer params file
            The optimizer params

        loss_function: loss file
            The loss function
        """

        total_param_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'\nTotal # of model\'s trainable parameters: {total_param_num:,}')
        print('----------------------------------------------------\n')

        self.model = model.to(self.device)
        self.loss_function = loss_function()
        self.optimiser = optimiser([
            {'params': self.model.parameters()},
            {'params': self.loss_function.parameters()}], **optimiser_params)

        # Saves the model, optimiser,loss function name for writing to config file
        self.model_info['optimiser'] = optimiser.__name__
        self.model_info['total_param_num'] = total_param_num
        self.model_info['loss_function'] = loss_function.__name__
        self.model_info['optimiser_params'] = optimiser_params
        self.model_info['num_epochs'] = self.num_epochs
        self.params['Network'] = self.model_info
        write_config(self.params, self.cfg_path,sort_keys=True)


    def load_checkpoint(self, model, optimiser, optimiser_params, loss_function):
        """In case of resuming training from a checkpoint,
        loads the weights for all the models, optimizers, and
        loss functions, and device, tensorboard events, number
        of iterations (epochs), and every info from checkpoint.

        Parameters
        ----------
        model: model file
            The network

        optimiser: optimizer file
            The optimizer

        optimiser_params: optimizer params file
            The optimizer params

        loss_function: loss file
            The loss function
        """
        checkpoint = torch.load(os.path.join(self.params['target_dir'],
                                             self.params['network_output_path'], self.params['checkpoint_name']))
        self.device = None
        self.model_info = checkpoint['model_info']
        self.setup_cuda()
        self.model = model.to(self.device)
        self.loss_function = loss_function()
        self.optimiser = optimiser([
            {'params': self.model.parameters()},
            {'params': self.loss_function.parameters()}], **optimiser_params)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.loss_function.load_state_dict(checkpoint['loss'])
        self.optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        self.writer = SummaryWriter(log_dir=os.path.join(os.path.join(
            self.params['target_dir'], self.params['tb_logs_path'])), purge_step=self.epoch + 1)


    def execute_training(self, train_loader, valid_loader=None, validiation=False):
        """Executes training by running training and validation at each iteration.
        This is the pipeline based on our own iteration-wise implementation.

        Parameters
        ----------
        validation: bool
            If we would like to do validation
       """
        self.params = read_config(self.cfg_path)
        total_start_time = time.time()

        for epoch in range(self.num_epochs - self.epoch):
            self.epoch += 1
            start_time = time.time()

            # train epoch
            train_loss = self.train_epoch(train_loader)

            # Validation epoch
            if (self.epoch) % self.params['valid_iteration_freq'] == 0:
                if validiation:
                    valid_loss, avg_EER, avg_FAR, avg_FRR, avg_thresh = self.valid_epoch(valid_loader)

            end_time = time.time()
            epoch_hours, epoch_mins, epoch_secs = self.time_duration(start_time, end_time)
            total_hours, total_mins, total_secs = self.time_duration(total_start_time, end_time)

            # saving the model, checkpoint, TensorBoard, etc.
            if (self.epoch) % self.params['valid_iteration_freq'] == 0:
                if validiation:
                    self.calculate_tb_stats(train_loss, valid_loss,
                                            avg_EER, avg_FAR, avg_FRR, avg_thresh)
                    self.savings_prints(epoch_hours, epoch_mins, epoch_secs, total_hours,
                                        total_mins, total_secs, train_loss, valid_loss,
                                        avg_EER, avg_FAR, avg_FRR, avg_thresh)
                else:
                    self.calculate_tb_stats(train_loss)
                    self.savings_prints(epoch_hours, epoch_mins, epoch_secs, total_hours,
                                        total_mins, total_secs, train_loss)


    def train_epoch(self, train_loader):
        """
        One iteration over all speakers; not all utterances
        will be chosen, but from every speaker some utterances will be chosen

        Parameters
        ----------
        train_loader: Pytorch dataloader object
            training data loader
        """
        self.model.train()
        total_loss = 0

        for idx, mel_db_batch in enumerate(train_loader):

            mel_db_batch = mel_db_batch.to(self.device)
            mel_db_batch = torch.reshape(mel_db_batch, (self.params['Network']['N'] * self.params['Network']['M'],
                                                        mel_db_batch.size(2), mel_db_batch.size(3)))
            # changing the utterances ordering (randomization)
            perm = random.sample(range(0, self.params['Network']['N'] * self.params['Network']['M']),
                                 self.params['Network']['N'] * self.params['Network']['M'])
            unperm = list(perm)
            for i, j in enumerate(perm):
                unperm[j] = i
            mel_db_batch = mel_db_batch[perm]

            self.optimiser.zero_grad()
            with torch.set_grad_enabled(True):
                embeddings = self.model(mel_db_batch)
                # changing the utterances ordering back
                embeddings = embeddings[unperm]
                embeddings = torch.reshape(embeddings, (self.params['Network']['N'],
                                                        self.params['Network']['M'], embeddings.size(1)))
                # (N, M, embedding)

                loss = self.loss_function(embeddings)
                loss.backward()

                # L2 normalization of gradient
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3.0)
                torch.nn.utils.clip_grad_norm_(self.loss_function.parameters(), 1.0)

                self.optimiser.step()
                total_loss = total_loss + loss

        epoch_loss = total_loss / (idx + 1)

        return epoch_loss



    def valid_epoch(self, valid_loader):
        """

        Parameters
        ----------
        valid_loader: Pytorch dataloader object
            validation data loader
        """

        self.model.eval()

        with torch.no_grad():
            avg_EER = 0
            avg_FAR = 0
            avg_FRR = 0
            avg_thresh = 0
            total_loss = 0
            for e in range(self.params['num_valid_epochs']):

                batch_loss = 0
                batch_avg_EER = 0
                batch_avg_FAR = 0
                batch_avg_FRR = 0
                batch_avg_thresh = 0
                for batch_id, mel_db_batch in enumerate(valid_loader):
                    assert self.params['Network']['M_valid'] % 2 == 0
                    mel_db_batch = mel_db_batch.to(self.device)
                    enrollment_batch, verification_batch = torch.split(mel_db_batch, int(mel_db_batch.size(1) / 2), dim=1)

                    # loss calculation for the whole data
                    mel_db_batch_l = torch.reshape(mel_db_batch, (self.params['Network']['N_valid'] *
                                                                  self.params['Network']['M_valid'], mel_db_batch.size(2), mel_db_batch.size(3)))
                    embeddings = self.model(mel_db_batch_l)
                    embeddings = torch.reshape(embeddings, (self.params['Network']['N_valid'], self.params['Network']['M_valid'], embeddings.size(1)))
                    # (N, M, embedding)
                    loss = self.loss_function(embeddings)
                    batch_loss += loss


                    # EER calculation with enrolment and verification
                    enrollment_batch = torch.reshape(enrollment_batch,
                                                     (self.params['Network']['N_valid'] * self.params['Network']['M_valid'] // 2,
                                                      enrollment_batch.size(2), enrollment_batch.size(3)))
                    verification_batch = torch.reshape(verification_batch,
                                                       (self.params['Network']['N_valid'] * self.params['Network']['M_valid'] // 2,
                                                        verification_batch.size(2), verification_batch.size(3)))
                    perm = random.sample(range(0, verification_batch.size(0)), verification_batch.size(0))
                    unperm = list(perm)
                    for i, j in enumerate(perm):
                        unperm[j] = i
                    verification_batch = verification_batch[perm]

                    enrollment_embeddings = self.model(enrollment_batch)
                    verification_embeddings = self.model(verification_batch)
                    verification_embeddings = verification_embeddings[unperm]
                    enrollment_embeddings = torch.reshape(enrollment_embeddings,
                                                          (self.params['Network']['N_valid'],
                                                           self.params['Network']['M_valid'] // 2, enrollment_embeddings.size(1)))
                    verification_embeddings = torch.reshape(verification_embeddings,
                                                            (self.params['Network']['N_valid'],
                                                             self.params['Network']['M_valid'] // 2, verification_embeddings.size(1)))
                    enrollment_centroids = get_centroids(enrollment_embeddings)
                    sim_matrix = get_cossim(verification_embeddings, enrollment_centroids)

                    # calculating EER
                    diff = 1
                    EER = 0
                    EER_thresh = 0
                    EER_FAR = 0
                    EER_FRR = 0
                    for thres in [0.01 * i + 0.0 for i in range(100)]:
                        sim_matrix_thresh = sim_matrix > thres
                        FAR = (sum(
                            [sim_matrix_thresh[i].float().sum() - sim_matrix_thresh[i, :, i].float().sum() for i in
                             range(int(self.params['Network']['N_valid']))]) / (self.params['Network']['N_valid'] - 1.0) / (float(self.params['Network']['M_valid'] / 2)) / self.params['Network']['N_valid'])
                        FRR = (sum([self.params['Network']['M_valid'] / 2 - sim_matrix_thresh[i, :, i].float().sum() for i in
                                    range(int(self.params['Network']['N_valid']))]) / (float(self.params['Network']['M_valid'] / 2)) / self.params['Network']['N_valid'])
                        # Save threshold when FAR = FRR (=EER)
                        if diff > abs(FAR - FRR):
                            diff = abs(FAR - FRR)
                            EER = (FAR + FRR) / 2
                            EER_thresh = thres
                            EER_FAR = FAR
                            EER_FRR = FRR
                    batch_avg_EER += EER
                    batch_avg_FAR += EER_FAR
                    batch_avg_FRR += EER_FRR
                    batch_avg_thresh += EER_thresh
                avg_EER += batch_avg_EER / (batch_id + 1)
                avg_FAR += batch_avg_FAR / (batch_id + 1)
                avg_FRR += batch_avg_FRR / (batch_id + 1)
                avg_thresh += batch_avg_thresh / (batch_id + 1)
                total_loss += batch_loss / (batch_id + 1)
                avg_thresh = torch.Tensor([avg_thresh]).to(self.device)
                avg_thresh = avg_thresh[0]

        epoch_loss = total_loss / self.params['num_valid_epochs']
        avg_EER = avg_EER / self.params['num_valid_epochs']
        avg_FAR = avg_FAR / self.params['num_valid_epochs']
        avg_FRR = avg_FRR / self.params['num_valid_epochs']
        avg_thresh = avg_thresh / self.params['num_valid_epochs']

        return epoch_loss, avg_EER, avg_FAR, avg_FRR, avg_thresh



    def savings_prints(self, epoch_hours, epoch_mins, epoch_secs, total_hours,
                       total_mins, total_secs, train_loss, valid_loss=None,
                       avg_EER=None, avg_FAR=None, avg_FRR=None, avg_thresh=None):
        """Saving the model weights, checkpoint, information,
        and training and validation loss and evaluation statistics.

        Parameters
        ----------
        iteration_hours: int
            hours part of the elapsed time of each iteration

        iteration_mins: int
            minutes part of the elapsed time of each iteration

        iteration_secs: int
            seconds part of the elapsed time of each iteration

        total_hours: int
            hours part of the total elapsed time

        total_mins: int
            minutes part of the total elapsed time

        total_secs: int
            seconds part of the total elapsed time

        train_loss: float
            training loss of the model

        valid_loss: float
            validation loss of the model

        avg_EER: float

        avg_FAR: float

        avg_FRR: float

        avg_thresh: float
        """

        # Saves information about training to config file
        self.params['Network']['num_steps'] = self.epoch
        write_config(self.params, self.cfg_path, sort_keys=True)

        # Saving the model based on reducing loss
        if valid_loss:
            if valid_loss < self.best_loss:
                self.best_loss = valid_loss
                torch.save(self.model.state_dict(), os.path.join(self.params['target_dir'], self.params['network_output_path']) + '/' +
                           self.params['trained_model_name'])
        else:
            if train_loss < self.best_loss:
                self.best_loss = train_loss
                torch.save(self.model.state_dict(), os.path.join(self.params['target_dir'], self.params['network_output_path']) + '/' +
                           self.params['trained_model_name'])

        # Saving every couple of epochs
        if (self.epoch) % self.params['network_save_freq'] == 0:
            torch.save(self.model.state_dict(), os.path.join(self.params['target_dir'], self.params['network_output_path']) + '/' +
                       'epoch{}_'.format(self.epoch) + self.params['trained_model_name'])

        # Save a checkpoint every "network_checkpoint_freq" epochs
        if (self.epoch) % self.params['network_checkpoint_freq'] == 0:
            torch.save({'epoch': self.epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimiser.state_dict(),
                        'loss': self.loss_function.state_dict(), 'num_epochs': self.num_epochs,
                        'model_info': self.model_info, 'best_loss': self.best_loss},
                       os.path.join(self.params['target_dir'], self.params['network_output_path']) + '/' + self.params['checkpoint_name'])

        print('\n------------------------------------------------------'
              '----------------------------------')
        print(f'Epoch: {self.epoch}/{self.num_epochs} | '
              f'Epoch Time: {epoch_hours}h {epoch_mins}m {epoch_secs}s | '
              f'Total Time: {total_hours}h {total_mins}m {total_secs}s')
        print(f'\n\tTrain Loss: {train_loss:.4f}')
        if valid_loss:
            print(f'\t Val. Loss: {valid_loss:.4f}')
            print(f'\tEER: {avg_EER * 100:.2f}% | FAR: {avg_FAR * 100:.2f}% | FRR: {avg_FRR * 100:.2f}% | Threshold: {avg_thresh:.2f}\n')

        # saving the training and validation stats
        if valid_loss:
            mesg = f'----------------------------------------------------------------------------------------\n' \
                   f'Epoch: {self.epoch}/{self.num_epochs} | Epoch Time: {epoch_hours}h {epoch_mins}m {epoch_secs}s' \
                   f' | Total Time: {total_hours}h {total_mins}m {total_secs}s\n\n\tTrain Loss: {train_loss:.4f}' \
                   f'\n\t Val. Loss: {valid_loss:.4f} | EER: {avg_EER * 100:.2f}% | FAR: {avg_FAR * 100:.2f}% | FRR: {avg_FRR * 100:.2f}% | Threshold: {avg_thresh:.2f}\n\n'
        else:
            mesg = f'----------------------------------------------------------------------------------------\n' \
                   f'Epoch: {self.epoch}/{self.num_epochs} | Epoch Time: {epoch_hours}h {epoch_mins}m {epoch_secs}s' \
                   f' | Total Time: {total_hours}h {total_mins}m {total_secs}s\n\n\tTrain Loss: {train_loss:.4f}\n\n'
        with open(os.path.join(self.params['target_dir'], self.params['stat_log_path']) + '/Stats', 'a') as f:
            f.write(mesg)


    def calculate_tb_stats(self, train_loss, valid_loss=None,
                           avg_EER=None, avg_FAR=None, avg_FRR=None, avg_thresh=None):
        """Adds the evaluation metrics and loss values to the tensorboard.

        Parameters
        ----------
        train_loss: float
            training loss of the model

        valid_loss: float
            validation loss of the model

        avg_EER: float

        avg_FAR: float

        avg_FRR: float

        avg_thresh: float
        """

        self.writer.add_scalar('Training' + '_Loss', train_loss, self.epoch)
        if valid_loss:
            self.writer.add_scalar('Validation' + '_Loss', valid_loss, self.epoch)
            self.writer.add_scalar('Validation' + '_EER', avg_EER, self.epoch)
            self.writer.add_scalar('Validation' + '_FAR', avg_FAR, self.epoch)
            self.writer.add_scalar('Validation' + '_FRR', avg_FRR, self.epoch)
            self.writer.add_scalar('Validation' + '_Threshold', avg_thresh, self.epoch)