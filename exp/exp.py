import torch
import pandas as pd
from collections import OrderedDict
import logging

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self._get_additional_features()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
    
    def _get_additional_features(self):
        float_features = []
        categorical_features = OrderedDict([
            ("month", 12),
            ("day", 31),
            ("weekday", 7),
            ("hour", 24),
        ])
        try:
            df_features = pd.read_csv(self.args.external_factors)
            df_float = df_features.select_dtypes(include=['float'])
            df_int = df_features.select_dtypes(include=['int'])

            float_features = df_float.columns
            logging.info("Additional float features: " + ", ".join(float_features))

            for col in df_int.columns:
                categorical_features[col] = len(df_int[col].unique())
        
        except Exception as e:
            logging.info("No additional features are used.")
        logging.info("Categorical features: " + ", ".join(categorical_features.keys()))

        self.args.float_features = float_features
        self.args.categorical_features = categorical_features

    def _acquire_device(self):
        if self.args.use_gpu:
            device = torch.device('cuda:{}'.format(self.args.gpu))
            logging.info('Using GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            logging.info('Using CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
