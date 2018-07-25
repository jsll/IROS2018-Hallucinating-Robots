import datetime
import torch as tc
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import data as hd
import plots as hp
import nets as hn

# Parameters
inf_replacement_n = 30
nan_replacement_n = 30
in_trim_sx = 199
in_trim_dx = 325
test_in_trim_sx = 210
test_in_trim_dx = 336
out_trim_sx = 20
out_trim_dx = 635
trim_length = 126

use_cuda = True

batch_sz = 32
n_of_epochs = 2000

in_file = '../dataset/input_data.txt'
out_file = '../dataset/output_data.txt'
test_in_file = '../dataset/input_test_data.txt'
test_out_file = '../dataset/output_test_data.txt'
net_dump_folder = 'nets/'

# Training set loading
trans_in = transforms.Compose([hd.Interval(in_trim_sx, in_trim_dx),
                               hd.TrimToLength(trim_length),
                               hd.ReplaceInf(inf_replacement_n),
                               hd.ReplaceNan(nan_replacement_n),
                               hd.Normalize(0,30)])
trans_out = transforms.Compose([hd.Interval(out_trim_sx, out_trim_dx),
                                hd.ScaleToLength(in_trim_dx-in_trim_sx),
                                hd.TrimToLength(trim_length),
                                hd.ReplaceInf(inf_replacement_n),
                                hd.ReplaceNan(nan_replacement_n),
                                hd.Normalize(0,30)])

train_dataset = hd.HallucinatingDataset(csv_in_file=in_file,
                                        csv_out_file=out_file,
                                        transform_in=trans_in,
                                        transform_out=trans_out)

train_dataloader = DataLoader(train_dataset, batch_size=batch_sz,
                              shuffle=True, num_workers=8)

# Test set loading
test_trans_in = transforms.Compose([hd.Interval(test_in_trim_sx, test_in_trim_dx),
                                    hd.TrimToLength(trim_length),
                                    hd.ReplaceInf(inf_replacement_n),
                                    hd.ReplaceNan(nan_replacement_n),
                                    hd.Normalize(0,30)])
test_trans_out = transforms.Compose([hd.Interval(out_trim_sx, out_trim_dx),
                                     hd.ScaleToLength(test_in_trim_dx-test_in_trim_sx),
                                     hd.TrimToLength(trim_length),
                                     hd.ReplaceInf(inf_replacement_n),
                                     hd.ReplaceNan(nan_replacement_n),
                                     hd.Normalize(0,30)])

test_dataset = hd.HallucinatingDataset(csv_in_file=test_in_file,
                                       csv_out_file=test_out_file,
                                       transform_in=test_trans_in,
                                       transform_out=test_trans_out)

test_dataloader = DataLoader(test_dataset, batch_size=batch_sz,
                             shuffle=True, num_workers=8)
