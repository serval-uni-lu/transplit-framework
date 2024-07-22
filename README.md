# Transplit: fast and cheaper energy demand forecasting

### Native installation

**Requirements**  
- Python **3.8+**
- Pytorch **1.10+**

**Optional**  
- PyTorch installed with a GPU to run it faster ;)

You can create a virtual environment with conda:
```shell
conda create -n transplit python=3.8
```

Install dependencies with:
```shell
pip install -r requirements.txt
```

### Docker usage
If you don't want to install it natively and want an easy solution, you can use Docker.

First pull the Nvidia's PyTorch image:
```shell
docker pull nvcr.io/nvidia/pytorch:23.02-py3
```

If you want to run the container with GPU, you will need to setup Docker for it
by installing the `nvidia-container-runtime` and `nvidia-container-toolkit` packages. 

Then run the container from this directory directory:
```shell
docker run -it --rm --gpus all --name transplit -v $(pwd):/workspace nvcr.io/nvidia/pytorch:23.02-py3
```

Once in it, install the dependencies with:
```shell
pip install -r requirements.txt
```
(already done in the Dockerfile, if you used it)

And download the data if not already done.

### Run one experiment
To run one experiment:
```shell
python run.py --data_path datasets/creos.csv --pred_len 168
```

### Dataset format
**Main dataset**
The dataset should be a csv file with named columns (on the first line), with the first column named `date`, containing the timestamps.
Every other column is considered as a separate consumption time series (e.g. a single consumer).
Example:
```csv
date,consumer1,consumer2,consumer3
2020-01-01 00:00:00,0.1,0.2,0.3
2020-01-01 00:15:00,0.2,0.3,0.4
2020-01-01 00:30:00,0.3,0.4,0.5
...
```

**External factors (optional)**
Some other features might be common to all consumers, such as the temperature, holidays, etc.
These features should be in a separate csv file, with the same format as the main dataset.
```csv
date,temperature,holiday
2020-01-01 00:00:00,10.0,0
2020-01-01 00:15:00,10.1,0
2020-01-01 00:30:00,10.2,0
...
```
**float** and **categorical** features are distinguished by their type (float or int).

**Note 1:** The timestamps should be the same in both files.
**Note 2:** For now, on this model, external factors are not efficient. Important changes to take them and the consumer's profile into account are currently under experiment for a new contribution.

### Model options
All options are listed in the `config.py` file. You can run the script with the `--help` option to see them.  
However, most of them are not intended to change; you might want to change the following ones:

- `--data_path`: dataset (csv) to use
- `--external_factors`: external factors dataset (csv) to use
- `--seq_len`: input sequence length (default: 336)
- `--pred_len`: prediction length (default: 168)
- `--period`: number of samples within one season (default: 24)
- `--batch_size`: batch size (default: 32)
  - *Note: modifying the batch size can alter the results; but in general, a higher batch size allows to train faster*
- `--train_ratio`: ratio of the dataset to use for training (default: 0.8 = 80%)
- `--test_ratio`: ratio of the dataset to use for testing (default: 0.2 = 20%)
  - *Note: the remaining ratio is used for validation, if needed*
- `--loss`: loss function to use (default: mse)
  - Other losses: `mae`, `huber`, [`adaptive`](https://openaccess.thecvf.com/content_CVPR_2019/papers/Barron_A_General_and_Adaptive_Robust_Loss_Function_CVPR_2019_paper.pdf)
  - *Note: if needed, you can easily implement your own loss in `exp/exp_main.py`, `_select_criterion` function*


### Reuse a trained model
```python
from models.transplit import Model
from config import get_config

args = get_config(
    pred_len=168,
    batch_size=32
)
model = Model(args)
model.load_model('checkpoints/path/model.pth')

# important: the model works with standardized data
output = model(
    'x_enc' = torch.rand(32, 336, 1),
    'x_mark_enc' = torch.zeros((32, 168, 4), dtype=int)
)
# output.shape: (32, 168, 1)
```

`x_mark_enc` has 4 channels: `month`, `day`, `weekday`, `hour`, respectively ranging from 0 to `11`, `30`, `6`, `23`.
Additional float or categorical features (if used during the training) should be concatenated to `x_enc` and `x_mark_enc` before passing it to the model.

## Acknowledgement

We acknowledge the following github repositories that made the base of our work:

https://github.com/zhouhaoyi/Informer2020  
https://github.com/AaltoML/generative-inverse-heat-dissipation  
https://github.com/BorealisAI/scaleformer  
https://github.com/jonbarron/robust_loss_pytorch.git  
https://github.com/Nixtla/neuralforecast  
