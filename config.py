import argparse
import torch
from exp.exp import ExpBase


parser = argparse.ArgumentParser(description='Transplit model for time series forecasting')

# basic config
parser.add_argument('--training', type=int, default=1, help='status')
parser.add_argument('--model', type=str, default='Transplit', help=('Transplit or Dedipeak'))

# data loader
parser.add_argument('--data_path', type=str, default='dataset/creos.csv', help='data file')
parser.add_argument('--external_factors', type=str, default='none',
                    help='external factors file (same length as the dataset)')
parser.add_argument('--period', type=int, default=24, help='duration separating each time series sample')
parser.add_argument('--freq', type=str, default='h',
                    help='actually not used, but kept for compatibility with the original code')
parser.add_argument('--no-shuffle', action='store_true', help='don\'t shuffle samples during training', default=False)
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='directory for model checkpoints')
parser.add_argument('--load_checkpoint', type=str, default='', help='checkpoint to load at the beginning')
parser.add_argument('--save_preds', action='store_true', help='save predictions', default=False)
parser.add_argument('--scale', action='store_true', default=False,
                help='whether the output data keeps the standardized scale for testing')

# forecasting task
parser.add_argument('--seq_len', type=int, default=336, help='input sequence length')
parser.add_argument('--pred_len', type=int, default=168, help='prediction sequence length')

# supplementary config for Transplit model
parser.add_argument('--n_filters', type=int, default=256, help='num of filters, for Transplit')

# supplementary config for Diffusion models
parser.add_argument('--num_steps', type=int, default=50, help='diffusion steps')
parser.add_argument('--condition_prob', type=float, default=0.8, help='condition probability during training')
parser.add_argument('--blur_sigma_min', type=float, default=0.5, help='Minimum blurring sigma')
parser.add_argument('--blur_sigma_max', type=float, default=16.0, help='Maximum blurring sigma')
parser.add_argument('--training_noise', type=float, default=0.01, help='noise factor for the forward process')
parser.add_argument('--sampling_noise', type=float, default=0.00, help='noise factor for the backward process')
parser.add_argument('--use_causal', action='store_true', default=False, help='use causal diffusion')
parser.add_argument('--transformer_model', type=str, default='', help='transformer model to use for the base forecast')
parser.add_argument('--transformer_checkpoint', type=str, default='', help='.pth file to load for the base forecast')
parser.add_argument('--transformer_blurring', type=int, default=7, help='num of blurring steps to apply to the transformer\'s output')
parser.add_argument('--transformer_deblurring', type=int, default=15, help='num of deblurring (backward) steps')

# model define
parser.add_argument('--enc_in', type=int, default=1, help='input channel size (default = 1 consumer)')
parser.add_argument('--c_out', type=int, default=1, help='output channel size')
parser.add_argument('--d_model', type=int, default=72, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=1, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers (non applicable to Transplit)')
parser.add_argument('--d_ff', type=int, default=256, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=3, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=0, help='number of data loader workers')
parser.add_argument('--itr', type=int, default=1, help='number of experiments')
parser.add_argument('--train_epochs', type=int, default=1, help='train epochs')
parser.add_argument('--train_ratio', type=float, default=0.8, help='train dataset ratio')
parser.add_argument('--val_ratio', type=float, default=0.0, help='validation dataset ratio')
parser.add_argument('--test_ratio', type=float, default=0.2, help='test dataset ratio')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience (number of epochs without improvement before stopping)')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='Exp', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')

# parse and add private args
args = parser.parse_args(namespace=argparse.Namespace(
    categorical=None,
    float_features=None,
))

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
if args.use_gpu:
    args.devices = '0'


def get_config(**kwargs) -> argparse.Namespace:
    """Updates the default config with keyword arguments."""
    config = argparse.Namespace(**args.__dict__)
    for k, v in kwargs.items():
        setattr(config, k, v)
    exp = ExpBase(config, build_model=False)
    return exp.args