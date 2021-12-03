"""
 @Time    : 2021/7/23 14:03
 @Author  : WuW15
"""
import os

import numpy as np
import argparse

from Predictor import ObjectBehaviorModel
from torch import optim
from data_builder import DataBuilder
from visualization import plot_loss
import torch
import sys, time
from datetime import datetime
from torch.optim.lr_scheduler import ExponentialLR as ExpLR
import joblib

MEMORY_LEN = 32  # batch size
FEATURE_NUM = 32
OBJ_NUM = 3
HEAD_NUM = 4
BLOCK_NUM = 2
CLASS_NUM = 4
REGRESSION_NUM = 3
TEST_DATA_RATIO = 0.2
TRAIN_DATA_RATIO = 0.7
BATCH_SIZE = 32
MAX_EPOCH = 10
PRINT_FOR_EVERY_BATCHES = 10
MAX_RECORD_LOSS = torch.tensor(80)
PATH_TO_RESULTS = './results/'


# mounted_input_path = sys.argv[1]
# print(mounted_input_path)
# print(os.listdir(mounted_input_path))


class ModelTrainer:
    def __init__(self, model, dataset_path,max_cycle=MAX_EPOCH, obj_num=OBJ_NUM):
        self.dataset_path = dataset_path +'/'
        self.max_cycle = max_cycle
        self.model = self._build_model(model)
        self.optimizer = self._prepare_optimizer()
        # self.data, self.label = self._prepare_data()
        self.obj_num = obj_num
        self.finish = False
        self.test_data = None
        self.test_label_cls_data = None
        self.test_label_reg_data = None
        self.valid_data = None
        self.valid_label_cls_data = None
        self.valid_label_reg_data = None
        self.train_data = None
        self.train_label_cls_data = None
        self.train_label_reg_data = None
        self.batch_size = BATCH_SIZE
        self.loss_fn_CE = torch.nn.CrossEntropyLoss(reduction='mean', weight=torch.tensor([2.0, 4.0, 4.0, 1.0]))
        self.loss_fn_MSE = torch.nn.MSELoss(reduction='mean')
        if not os.path.exists(PATH_TO_RESULTS):
            os.makedirs(PATH_TO_RESULTS)
        print('testtesttest')
        print(self.dataset_path)

    @staticmethod
    def _build_model(model):
        m = model(dim=FEATURE_NUM,
                  num_obj=OBJ_NUM,
                  num_head=HEAD_NUM,
                  num_block=BLOCK_NUM,
                  num_cls=CLASS_NUM,
                  num_reg=REGRESSION_NUM,
                  hist_len=MEMORY_LEN)
        return m

    def _prepare_optimizer(self):
        optimizer = optim.Adam(self.model.parameters(), betas=(0.9, 0.999), eps=1e-8, lr=1e-4)
        return optimizer

    def _prepare_data(self):
        data_builder = DataBuilder(self.dataset_path)
        # all_data, *all_label_xxx_data = data_builder.build_random(clip_num=20, batch_size=10)
        print("################ Building Data ##################")
        all_data, all_label_cls_data, all_label_reg_data, pos_case_num = data_builder.build()
        all_data = all_data.permute(0, 2, 1)
        len_all_data = len(all_data)
        neg_case_num = len_all_data - pos_case_num
        # cut data in to train/test/validation according to their labels
        train_pos_idx = np.arange(0, int(pos_case_num * TRAIN_DATA_RATIO))
        test_pos_idx = np.arange(int(pos_case_num * TRAIN_DATA_RATIO),
                                 int(pos_case_num * (TRAIN_DATA_RATIO + TEST_DATA_RATIO)))
        valid_pos_idx = np.arange(int(pos_case_num * (TRAIN_DATA_RATIO + TEST_DATA_RATIO)), pos_case_num)

        train_neg_idx = np.arange(pos_case_num, int(neg_case_num * TRAIN_DATA_RATIO))
        test_neg_idx = np.arange(int(neg_case_num * TRAIN_DATA_RATIO),
                                 int(neg_case_num * (TRAIN_DATA_RATIO + TEST_DATA_RATIO)))
        valid_neg_idx = np.arange(int(neg_case_num * (TRAIN_DATA_RATIO + TEST_DATA_RATIO)), len_all_data)

        train_idx = np.concatenate((train_pos_idx, train_neg_idx))
        test_idx = np.concatenate((test_pos_idx, test_neg_idx))
        valid_idx = np.concatenate((valid_pos_idx, valid_neg_idx))

        self.valid_data = all_data[valid_idx]
        self.valid_label_cls_data = all_label_cls_data[valid_idx]
        self.valid_label_reg_data = all_label_reg_data[valid_idx]

        self.test_data = all_data[test_idx]
        self.test_label_cls_data = all_label_cls_data[test_idx]
        self.test_label_reg_data = all_label_reg_data[test_idx]

        self.train_data = all_data[train_idx]
        self.train_label_cls_data = all_label_cls_data[train_idx]
        self.train_label_reg_data = all_label_reg_data[train_idx]

    @staticmethod
    def _loss_func(func, ipt, tgt):
        return func(ipt, tgt)

    def run_model_test(self):
        src = torch.rand(1, MEMORY_LEN, FEATURE_NUM*OBJ_NUM)
        import time
        s = time.time()
        for i in range(0, 100):
            out = self.model(src)
            print(out)
        e = time.time()
        du = e - s
        print("single inference {}s".format(du / 100))

    def run_once_valid(self, frame, *label):
        with torch.no_grad():
            label_c, label_r = label
            test_output = torch.zeros(self.obj_num, CLASS_NUM)
            frame = torch.reshape(frame, (1, frame.shape[0], frame.shape[1]))
            test_output[0, :], test_output[1, :], test_output[2, :], batch_reg_out = \
                self.model(frame)
            clip_output_mean = test_output#.permute(1, 0)
            loss = torch.sum(self._loss_func(self.loss_fn_CE, clip_output_mean, label_c)) + \
                   torch.sum(self._loss_func(self.loss_fn_MSE, batch_reg_out.squeeze(), label_r))
        return loss

    def inference(self, data_clip):
        self.model.eval()
        output_cls = torch.zeros(self.obj_num, CLASS_NUM)
        data_clip = torch.reshape(data_clip, (1, data_clip.shape[0], data_clip.shape[1]))
        output_cls[0, :], output_cls[1, :], output_cls[2, :], output_reg = \
            self.model(data_clip)
        return output_cls, output_reg

    # def next_frame(self):
    #     frame = []
    #     label = dict()
    #     if self.data:
    #         frame = self.data[0]
    #         label = self.label[0]
    #     else:
    #         pass
    #     return frame, label

    # def run(self):
    #     while not self.finish:
    #         frame,label = self.next_frame()
    #         self.run_once_valid(frame, label)

    def train(self):
        # f = open("log{}.txt".format(time.time()), 'w')
        ######################
        # BUILD DATA
        ######################
        self._prepare_data()
        ######################
        # BUILD OPTIMIZER
        ######################
        self.optimizer.zero_grad()
        scheduler = ExpLR(self.optimizer, gamma=0.8)
        ######################
        # Training
        ######################
        clip_num = len(self.train_data)
        train_loss_rec = []
        eval_loss_rec = []
        try:
            for cycle in range(self.max_cycle):
                train_loss = 0
                # BATCH: train_data[batch_start: batch_end]
                batch_start = 0
                train_data_idx = np.arange(clip_num, dtype=int)
                np.random.shuffle(train_data_idx)
                # shuffle train data for each epoch
                train_data = self.train_data[train_data_idx]
                train_label_cls_data = self.train_label_cls_data[train_data_idx]
                train_label_reg_data = self.train_label_reg_data[train_data_idx]
                print_counter = 0
                while batch_start < clip_num:
                    batch_end = batch_start + self.batch_size
                    batch_size = self.batch_size
                    if batch_end > clip_num:
                        batch_end = clip_num
                        batch_size = batch_end - batch_start
                    # batch_reg_label = torch.zeros([32, 1, 1], dtype=torch.float)
                    # batch_cls_out = torch.zeros(32, 3, 3)
                    batch_cls_out = torch.zeros(batch_size, self.obj_num, CLASS_NUM)
                    train_data_batch = train_data[batch_start: batch_end]
                    # for j in range(batch_size):
                    batch_cls_out[:, 0, :], batch_cls_out[:, 1, :], batch_cls_out[:, 2, :], \
                        batch_reg_out = self.model(train_data_batch)
                    loss_CE = torch.tensor(0.)
                    loss_MSE = torch.tensor(0.)
                    for i, batch_idx in enumerate(np.arange(batch_start, batch_end)):
                        loss_CE += self._loss_func(self.loss_fn_CE,
                                                   batch_cls_out[i], train_label_cls_data[batch_idx])
                        # loss_MSE += self._loss_func(self.loss_fn_MSE,
                        #                             batch_reg_out[i].squeeze(), train_label_reg_data[batch_idx])
                    loss_CE_avg = self.obj_num * loss_CE / batch_size  # * self.obj_num)
                    loss_MSE_avg = loss_MSE / batch_size  # * self.obj_num)
                    loss = loss_CE + loss_MSE
                    loss_avg = loss_CE_avg + loss_MSE_avg
                    # train_loss += loss_CE
                    # record the most recent batch average loss
                    train_loss = loss_avg
                    loss_avg.backward(retain_graph=False)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                    if print_counter % PRINT_FOR_EVERY_BATCHES == 0:
                        print("################ EPOCH ", int(cycle), " ##################")
                        print("################ STEP ", int(batch_end - 1), " ##################")
                        # print('training output class: ', batch_cls_out[-1])
                        # print('training output regression: ', batch_reg_out[-1])
                        # print('training label class:', train_label_cls_data[batch_end - 1])
                        # print('training label regression:', train_label_reg_data[batch_end - 1])
                        print('training loss_CE: ', loss_CE_avg)
                        print('training loss_MSE: ', loss_MSE_avg)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    batch_start += batch_size
                    print_counter += 1
                # train_loss /= len(train_data)
                train_loss_rec.append(train_loss)
                self.model.eval()
                print("################ EPOCH END ##################")
                print("################ VALIDATION ##################")
                eval_loss = self.eval()
                print('training loss: ', train_loss)
                print('validation loss: ', eval_loss)
                eval_loss_rec.append(eval_loss)
                self.model.train()
                scheduler.step()
        except Exception as e:
            current_time_str = "_{:d}_{:d}_{:d}_{:d}_{:d}_{:d}".format(datetime.now().year, datetime.now().month,
                                                                       datetime.now().day, datetime.now().hour,
                                                                       datetime.now().minute, datetime.now().second)
            torch.save(self.model, 'model.pt')
            #os.makedirs('thisoutputs', exist_ok=True)
            #joblib.dump(value=self.model, filename='thisoutputs/thismodel.pt')
            torch.save(train_loss_rec, '{}/train_loss_rec{}'.format(PATH_TO_RESULTS, current_time_str))
            torch.save(eval_loss_rec, '{}/eval_loss_rec{}'.format(PATH_TO_RESULTS, current_time_str))
            raise e
        # save model before exit
        current_time_str = "_{:d}_{:d}_{:d}_{:d}_{:d}_{:d}".format(datetime.now().year, datetime.now().month,
                                                                   datetime.now().day, datetime.now().hour,
                                                                   datetime.now().minute, datetime.now().second)
        torch.save(self.model, 'model.pt')
        os.makedirs('outputs', exist_ok=True)
        joblib.dump(value=self.model, filename='outputs/thismodel.pt')
        #torch.save(self.model, os.path.join(args.output_dir, 'model.pt'))
        torch.save(train_loss_rec, '{}/train_loss_rec{}'.format(PATH_TO_RESULTS, current_time_str))
        torch.save(eval_loss_rec, '{}/eval_loss_rec{}'.format(PATH_TO_RESULTS, current_time_str))
        # f.close()
        return current_time_str

    def eval(self):
        """
            Return average loss of test dataset.

        Returns:

        """
        test_data = self.test_data
        test_label_cls_data = self.test_label_cls_data
        test_label_reg_data = self.test_label_reg_data
        # test_data = self.train_data
        # test_label_cls_data = self.train_label_cls_data
        # test_label_reg_data = self.train_label_reg_data
        loss = 0.
        for test_d, test_l, test_r in zip(test_data, test_label_cls_data, test_label_reg_data):
            #     loss += self.run_once_valid(test_d, test_l, test_r)
            with torch.no_grad():
                test_cls_output = torch.zeros(self.obj_num, CLASS_NUM)
                test_d = torch.reshape(test_d, (1, test_d.shape[0], test_d.shape[1]))
                test_cls_output[0, :], test_cls_output[1, :], test_cls_output[2, :], batch_reg_out = \
                    self.model(test_d)
                loss_CE = self.obj_num * self._loss_func(self.loss_fn_CE, test_cls_output, test_l)
                loss_MSE = torch.tensor(0.)
                # loss_MSE = self._loss_func(self.loss_fn_MSE, batch_reg_out.squeeze(), test_r)
                loss += loss_CE + loss_MSE
        return loss / len(test_data)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='data folder mounting point')
    parser.add_argument('-f')
    args = parser.parse_args()
    print(args)

    print("DATA PATH: " + args.data_path)
    print("LIST FILES IN DATA PATH...")
    print(os.listdir(args.data_path))
    Trainer = ModelTrainer(ObjectBehaviorModel,args.data_path)
    # Trainer.run_model_test()
    # current_time_str = "_2021_9_27_12_22_52"
    current_time_str = Trainer.train()
    train_loss_rec = torch.load('{}/train_loss_rec{}'.format(PATH_TO_RESULTS, current_time_str))
    eval_loss_rec = torch.load('{}/eval_loss_rec{}'.format(PATH_TO_RESULTS, current_time_str))
    train_loss_rec = [i.detach().numpy() for i in train_loss_rec]
    eval_loss_rec = [i.detach().numpy() for i in eval_loss_rec]
    # train_loss_rec.pop(0)
    # eval_loss_rec.pop(0)
    plot_loss(train_loss_rec, eval_loss_rec)
    # load the trained model and inference
    # model = torch.load('model.pt')
    # model.eval()

