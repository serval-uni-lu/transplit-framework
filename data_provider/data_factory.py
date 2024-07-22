from data_provider.data_loader import ElectricityDataset, DatesBatchSampler
from torch.utils.data import DataLoader
import pandas as pd


def data_provider(args, flag):
    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
    else:
        shuffle_flag = not args.no_shuffle
        drop_last = True
        batch_size = args.batch_size
    
    df = pd.read_csv(args.data_path)

    data_set = ElectricityDataset(
        df=df,
        flag=flag,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        split=(args.train_ratio, args.val_ratio, args.test_ratio),
        add_features=args.external_factors,
        period=args.period
    )
    print(flag, "size:", len(data_set))

    # batch_sampler = DatesBatchSampler(data_set, batch_size, drop_last)
    #
    # data_loader = DataLoader(
    #     data_set,
    #     # batch_size=batch_size,
    #     batch_sampler=batch_sampler,
    #     # shuffle=shuffle_flag,
    #     num_workers=args.num_workers,
    #     # drop_last=drop_last
    # )
    
    if len(data_set) > 0:
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last
        )
    else:
        data_loader = []
    return data_set, data_loader
