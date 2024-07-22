from data_provider.data_factory import data_provider
from exp.exp import ExpBase
from models import transplit
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from robust_loss_pytorch import AdaptiveLossFunction

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import os
import time
import math
import warnings
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


class ExpTransplit(ExpBase):
    def __init__(self, args):
        super(ExpTransplit, self).__init__(args)

    def _build_model(self):
        model = transplit.Model(self.args).float()
        logging.info(f"NUMBER OF PARAMETERS IN MODEL: {self.args.model}: {sum(p.numel() for p in model.parameters())}")
        if self.args.load_checkpoint:
            model.load_state_dict(torch.load(self.args.load_checkpoint))
            logging.info(f"Model loaded from {self.args.load_checkpoint}")
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_loader

    def _select_optimizer(self):
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


    def vali(self, vali_loader, criterion, save=False):
        total_loss = []
        X = []
        yt = []
        yp = []
        means = []
        scales = []
        self.model.eval()
        with torch.no_grad():
            for i, batch in tqdm(enumerate(vali_loader)):
                batch_x = batch["seq_x"].float().to(self.device)
                batch_y = batch["seq_y"].float().to(self.device)
                batch_x_mark = batch["seq_x_mark"].to(self.device)
                batch_y_mark = batch["seq_y_mark"].to(self.device)
                scale = batch["scale"]
                mean = batch["mean"]

                # encoder - decoder
                outputs = self.model(
                    x_enc=batch_x,
                    x_mark_enc=batch_x_mark,
                    x_dec=None,
                    x_mark_dec=batch_y_mark
                )

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                if save:
                    X.append(batch_x.detach().cpu())
                    yt.append(true)
                    yp.append(pred)
                    means.append(mean)
                    scales.append(scale)

                assert pred.shape==true.shape
                loss = criterion(pred, true)
                total_loss.append(loss.flatten())
        
        total_loss = torch.cat(total_loss, dim=0).mean().item()

        if save:
            X = np.concatenate(X, axis=0)
            yt = np.concatenate(yt, axis=0)
            yp = np.concatenate(yp, axis=0)
            means = np.concatenate(means, axis=0)
            scales = np.concatenate(scales, axis=0)
            np.savez(
                os.path.join(self.args.checkpoints, 'preds.npz'),
                X=X, yt=yt, yp=yp, means=means, scales=scales
            )
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
        logging.info("total iterations:", train_steps)
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

                # encoder - decoder
                outputs = self.model(
                    x_enc=batch_x,
                    x_mark_enc=batch_x_mark,
                    x_dec=None,
                    x_mark_dec=batch_y_mark
                )

                assert outputs.shape==batch_y.shape, f"{outputs.shape}, {batch_y.shape}"

                if self.args.loss == 'adaptive':
                    loss = criterion((outputs - batch_y).flatten().unsqueeze(-1))
                else:
                    loss = criterion(outputs, batch_y)
                loss = loss.mean()
                train_loss.append(loss.item())

                if (i + 1) % 1000 == 0:
                    logging.info("iters: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - t0) / 1000
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    logging.info('  speed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    t0 = time.time()

                loss.backward()
                optim.step()
                if self.args.loss == 'adaptive':
                    criterion.optim.step()

            logging.info("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            if len(vali_loader) > 0:
                logging.info("Validation")
                vali_loss = self.vali(vali_loader, criterion)
                early_stopping(vali_loss, self.model, path)
                vali_loss = round(vali_loss, 7)
            else:
                torch.save(self.model.state_dict(), path + '/' + f'checkpoint.pth')
                logging.info("checkpoint saved")
                vali_loss = None

            logging.info("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3}".format(
                epoch + 1, train_steps, train_loss, vali_loss
            ))
            if early_stopping.early_stop:
                logging.info("Early stopping")
                break

            adjust_learning_rate(optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model


    def test(self, setting, checkpoint="",):
        test_loader = self._get_data(flag='test')
        
        if checkpoint:
            logging.info('loading model')
            self.model.load_state_dict(torch.load(checkpoint))

        X = []
        preds = []
        trues = []
        means = []
        scales = []

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, batch in (pbar := tqdm(enumerate(test_loader), total=len(test_loader))):
                batch_x = batch["seq_x"].float().to(self.device)
                batch_y = batch["seq_y"].float().to(self.device)
                batch_x_mark = batch["seq_x_mark"].to(self.device)
                batch_y_mark = batch["seq_y_mark"].to(self.device)
                scale = batch["scale"]
                mean = batch["mean"]

                # encoder - decoder
                outputs = self.model(
                    x_enc=batch_x,
                    x_mark_enc=batch_x_mark,
                    x_dec=None,
                    x_mark_dec=batch_y_mark
                )

                outputs = outputs.detach().cpu()
                batch_y = batch_y.detach().cpu()
                if not self.args.scale:
                    outputs = outputs * scale + mean
                    batch_y = batch_y * scale + mean

                pred = outputs.numpy()
                true = batch_y.numpy()

                preds.append(pred)
                trues.append(true)

                if (i + 1) % 1000 == 0:
                    loss = np.mean((np.vstack(preds[-1000:]) - np.vstack(trues[-1000:])) ** 2)
                    pbar.set_description("Test loss: {}".format(loss))

                if self.args.save_preds:
                    X.append(batch_x.detach().cpu().numpy())
                    means.append(mean.detach().cpu().numpy())
                    scales.append(scale.detach().cpu().numpy())

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
            if self.args.features != 'SA':
                means = means[0]
                scales = scales[0]
            np.savez(
                folder_path + 'preds.npz',
                X=X, preds=preds, trues=trues, means=means, scales=scales
            )
        logging.info('test shape:', preds.shape, trues.shape)

        mae, mse, rmse, mape, mspe, pmse, pmae = metric(preds, trues)
        logging.info('mse:{}, mae:{}, pmae:{}'.format(mse, mae, pmae))

        results_dict = {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "mape": mape,
            "mspe": mspe,
            "pmse": pmse,
            "pmae": pmae
        }

        return results_dict