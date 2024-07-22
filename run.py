
import json
import torch
from config import args
from exp.exp_main import ExpTransplit
from exp.exp_diffusion import ExpDiffusion
import random
import time
import numpy as np


fix_seed = 2024
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


if __name__ == "__main__":
    print('Args in experiment:')
    print(args)

    if args.model in ['Dedipeak']:
        Exp = ExpDiffusion
    else:
        Exp = ExpTransplit

    if args.training:
        for ii in range(args.itr):
            timecode = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
            setting = f'{args.model}_{args.data_path.split("/")[-1][:-4]}_{timecode}'
            exp = Exp(args)  # set experiments
            print('---- training: {} ----'.format(setting))
            exp.train(setting)

            print('---- testing: {} ----'.format(setting))
            exp.test(setting)

            torch.cuda.empty_cache()
    else:
        for ii in range(args.itr):
            setting = f'{args.data_path[:-4]}_{args.model}_{args.pred_len}_{args.loss}_{ii}'

            exp = Exp(args)  # set experiments
            print('---- testing: {} ----'.format(setting))
            results = exp.test(setting, f"./checkpoints/{setting}/checkpoint.pth")

            torch.cuda.empty_cache()