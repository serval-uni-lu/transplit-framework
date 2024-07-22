from argparse import Namespace
from data_provider.data_factory import data_provider
from exp.exp import Exp_Basic
from exp.exp_main import ExpTransplit
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric, all_peak_metrics
from robust_loss_pytorch import AdaptiveLossFunction
from models import deblurring
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from random import random
import os
import time
import warnings
import logging

warnings.filterwarnings('ignore')


class ExpDiffusion(Exp_Basic):
    def __init__(self, args):
        super(ExpDiffusion, self).__init__(args)

    def _build_model(self):
        model = deblurring.Model(self.args, self.device).float()
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_loader

    def _select_optimizer(self, additional_params=None):
        optim = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        return optim

    def _select_criterion(self):
        if self.args.loss == 'huber':
            criterion = torch.nn.HuberLoss(reduction='none', delta=0.5)
        elif self.args.loss == 'l1' or self.args.loss == 'mae':
            criterion = torch.nn.L1Loss(reduction='none')
        elif self.args.loss == 'adaptive':
            adaptive = AdaptiveLossFunction(1, torch.float32, self.device, alpha_hi=3.0)
            criterion = adaptive.lossfun
            criterion.optim = torch.optim.AdamW(list(adaptive.parameters()), lr=0.001)
        else: # default = MSE loss
            criterion = torch.nn.MSELoss(reduction='none')
        return criterion

    def vali(self, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, batch in tqdm(enumerate(vali_loader)):
                batch_x = batch["seq_x"].float().to(self.device)
                batch_y = batch["seq_y"].float().to(self.device)
                batch_x_mark = batch["seq_x_mark"].to(self.device)
                batch_y_mark = batch["seq_y_mark"].to(self.device)

                t = self.model.get_t().to(self.device)
                dec_inp, eps = self.model.q_xt_x0(batch_y, t)
                eps_hat = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, t)

                pred = eps_hat.detach().cpu()
                true = eps.detach().cpu()

                assert pred.shape == true.shape, f"pred shape: {pred.shape}, true shape: {true.shape}"
                loss = criterion(pred, true)
                total_loss.append(loss.flatten())
        total_loss = torch.cat(total_loss, dim=0).mean().item()
        self.model.train()
        return total_loss

    def train(self, setting):
        train_loader = self._get_data(flag='train')
        vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        t0 = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, batch in enumerate(train_loader):
                optim.zero_grad()
                if self.args.loss == 'adaptive':
                    criterion.optim.zero_grad()
                batch_x = batch["seq_x"].float().to(self.device)
                batch_y = batch["seq_y"].float().to(self.device)
                batch_x_mark = batch["seq_x_mark"].to(self.device)
                batch_y_mark = batch["seq_y_mark"].to(self.device)

                t = self.model.get_t().to(self.device)
                dec_inp, eps = self.model.q_xt_x0(batch_y, t)

                # sometimes, condition on the input (for classifier-free guidance at inference)
                if random() < self.args.condition_prob:
                    eps_hat = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, t)
                else:
                    eps_hat = self.model(None, None, dec_inp, batch_y_mark, t)

                assert eps_hat.shape == eps.shape, f"{eps_hat.shape=}, {eps.shape=}"

                if self.args.loss == 'adaptive':
                    loss =  criterion((eps_hat - eps).flatten().unsqueeze(-1))
                else:
                    loss =  criterion(eps_hat, eps)
                loss = loss.mean()
                if self.args.loss == 'mse':
                    loss = loss * 100
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    logging.info("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - t0) / 100
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    logging.info('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    t0 = time.time()
                    
                loss.backward()
                optim.step()
                if self.args.loss == 'adaptive':
                    optim.step()

            logging.info("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            if len(vali_loader) > 0:
                logging.info("Validation")
                vali_loss = self.vali(vali_loader, criterion)
                early_stopping(vali_loss, self.model, path)
            else:
                vali_loss = None

            logging.info("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss
            ))
            if early_stopping.early_stop:
                logging.info("Early stopping")
                break

            adjust_learning_rate(optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_loader = self._get_data(flag='test')
        if test:
            logging.info('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        running_times = []
        preds = []
        trues = []
        with torch.no_grad():
            for i, batch in (pbar := tqdm(enumerate(test_loader), total=len(test_loader))):
                batch_x = batch["seq_x"].float().to(self.device)
                batch_y = batch["seq_y"].float().to(self.device)
                batch_x_mark = batch["seq_x_mark"].to(self.device)
                batch_y_mark = batch["seq_y_mark"].to(self.device)
                scale = batch["scale"]
                mean = batch["mean"]

                start_time = time.time()

                t = self.model.get_t().to(self.device)
                dec_inp, eps = self.model.q_xt_x0(batch_y, t)

                eps_hat = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, t)

                running_times.append(time.time() - start_time)
                
                eps = eps.detach().cpu()
                eps_hat = eps_hat.detach().cpu()
                batch_x = batch_x.detach().cpu()
                batch_y = batch_y.detach().cpu()

                pred = eps_hat.numpy()
                true = eps.numpy()

                preds.append(pred)
                trues.append(true)

                if (i + 1) % 1000 == 0:
                    loss = np.mean((np.vstack(preds) - np.vstack(trues)) ** 2)
                    pbar.set_description("Test loss: {}".format(loss))

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        if self.args.save_preds:
            X = np.concatenate(X, axis=0)
            means = np.concatenate(means, axis=0)
            scales = np.concatenate(scales, axis=0)
            np.savez(
                folder_path + 'preds.npz',
                X=X, preds=preds, trues=trues, means=means, scales=scales
            )
        logging.info('test shape:', preds.shape, trues.shape)

        mae, mse, rmse, mape, mspe, pmse, pmae = metric(preds, trues)
        logging.info('mse:{}, mae:{}'.format(mse, mae))
        logging.info(f'running time: {np.array(running_times).sum()}')
        results_dict = {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "pmse": pmse,
            "pmae": pmae
        }
        return results_dict

    def predict(self, setting, load=False):
        pred_loader = self._get_data(flag='test')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))
        
        targs = Namespace(**self.args.__dict__)
        targs.model = self.args.transformer_model
        texp = ExpTransplit(targs)
        logging.info('loading transformer model:', self.args.transformer_checkpoint)
        texp.model.load_state_dict(torch.load(self.args.transformer_checkpoint))
        texp.model.eval()

        inputs = []
        trues = []
        tpreds = []
        preds = []

        mses = []
        p3sws = []
        p3eus = []

        self.model.eval()
        with torch.no_grad():
            for j, batch in (pbar := tqdm(enumerate(pred_loader), total=len(pred_loader))):
                batch_x = batch["seq_x"].float().to(self.device)
                batch_y = batch["seq_y"].float().to(self.device)
                batch_x_mark = batch["seq_x_mark"].to(self.device)
                batch_y_mark = batch["seq_y_mark"].to(self.device)
                scale = batch["scale"]
                mean = batch["mean"]

                x_dec = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                x_dec = torch.cat([batch_y[:, :self.args.label_len, :], x_dec], dim=1).float().to(self.device)
                outputs = texp.model(
                    batch_x[:, -targs.seq_len:, :],
                    batch_x_mark[:, -targs.seq_len:, :],
                    x_dec,
                    batch_y_mark
                )
                tpreds.append(outputs.detach().cpu().numpy())
                self.model.batch_size = self.args.batch_size
                t = self.model.get_t(self.args.transformer_blurring).to(self.device)
                dec_inp, _ = self.model.q_xt_x0(outputs, t)
                
                batch_y = batch_y[:, -self.args.pred_len:]
                batch_y_mark = batch_y_mark[:, -self.args.pred_len:]

                pred = []
                channels = 32
                for i in range(0, self.args.c_out, channels):
                    subset = slice(i, i+channels)
                    b = batch_x[..., subset].shape[-1]
                    self.model.batch_size = self.args.batch_size * b

                    for outputs in self.model.iterate_through_inference(
                        batch_x[..., subset].transpose(1, 2).reshape(-1, self.args.seq_len, 1),
                        batch_x_mark.repeat(b, 1, 1),
                        dec_inp[..., subset].transpose(1, 2).reshape(-1, self.args.pred_len, 1),
                        batch_y_mark.repeat(b, 1, 1),
                        steps=self.args.transformer_deblurring
                    ):
                        # possible to execute code during the deblurring here
                        pass
                    
                    pred.append(outputs.detach().reshape(-1, b, self.args.pred_len).transpose(1, 2).cpu().numpy())
                pred = np.concatenate(pred, axis=-1)

                inputs.append(batch_x.detach().cpu().numpy())
                trues.append(batch_y.detach().cpu().numpy())
                preds.append(pred)

                if (j + 1) % 1000 == 0:
                    loss = np.mean((np.vstack(preds) - np.vstack(trues)) ** 2)
                    pbar.set_description("Test loss: {}".format(loss))

        inputs = np.concatenate(inputs, axis=0)
        trues = np.concatenate(trues, axis=0)
        tpreds = np.concatenate(tpreds, axis=0)
        preds = np.concatenate(preds, axis=0)

        # mse
        mse = np.mean((preds - trues) ** 2)
        # mae
        mae = np.mean(np.abs(preds - trues))
        logging.info('mse:{}, mae:{}'.format(mse, mae))

        # result save
        folder_path = './results/' + self.args.transformer_setting \
            + f"_tnoise{self.args.training_noise}_snoise{self.args.sampling_noise}" + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # np.savez(folder_path + 'real_prediction.npz', inputs=inputs, preds=preds, trues=trues)
        # np.savez(folder_path + 'real_prediction.npz', inputs=inputs, tpreds=tpreds, preds=preds, trues=trues)

        avg_mse = np.mean(mses, axis=0)
        avg_p3sw = np.mean(p3sws, axis=0)
        avg_p3eu = np.mean(p3eus, axis=0)

        results_dict = {
            "mse": mse,
            "mae": mae,
            "avg_mse": avg_mse,
            "avg_p3sw": avg_p3sw,
            "avg_p3eu": avg_p3eu
        }
        return