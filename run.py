import torch
import numpy as np
from matplotlib import pyplot as plt
import torch.nn as nn
from torch import cuda
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from random import shuffle
import pickle
from time import time
# from torch._six import int_classes as _int_classes
from os import system

from data import *
from models import *
from utils import *

device = torch.device('cuda') if cuda.is_available() else torch.device('cpu')
print('Device is', device)

# CNNFold
model_address = 'models/cnnfold.mdl'

# BLOSSOM for the post-processing. If it's False, argmax would be used.
use_blossom = False

seq_to_family_file = 'datasets/seq_to_family.pkl'
train_dataset = 'datasets/align_train.csv'
test_dataset = 'datasets/align_test.csv'
start_point_in_csv = 0

model = ResNet()

checkpoint = torch.load(model_address)
model.load_state_dict(checkpoint['model_state_dict'])

cnn = model.to(device)

def seq_to_representation(seq):
    return create_matrix(seq, onehot=True, min_dist=3)

def input_tensor_to_target_matrix(tensor):
    return None

def target_matrix_to_structure(tensor):
    return None

###############################################################################
#                                  "testing"                                  #
###############################################################################

test_samples = pickle.load(open('test_list', 'rb'))

test_loader = DataLoader(test_samples,
                         shuffle=True,
                         batch_size=1,
                         collate_fn=lambda b:collate(b, onehot_inputs))

for i, (x, y, family) in enumerate(test_loader):

    x = x.to(device) # [B, 1, N, N]
    y = y.to(device) # [B, 1, N, N]

    y_hat_test = cnn(x, test=True, repeat=prediction_repeat) # [B, 1, N, N]

    mask = (y != -1).float() # [B, 1, N, N]
    y_hat_test = y_hat_test * mask # [B, 1, N, N]
    y_hat_test = binarize(y_hat_test, threshold, use_blossom=use_blossom)

    pk_res += f1_pk(y_hat_test, y)

    f1, prec, rec = calculate_f1(y_hat_test, y, None, is_2d=is_2d,
                                 consider_unpairings=consider_unpairings,
                                 reduction=False,
                                 shift_allowed=shift_allowed,
                                 prec_rec=True,
                                 only_pk=only_pk)

    # if only_pk:
    #     f1 = recall(y_hat_test, y, None, reduction=False, shift_allowed=shift_allowed,
    #         consider_unpairings=consider_unpairings, is_2d=is_2d, only_pk=only_pk)

    f1 = list(np.array(f1.cpu()))
    prec = list(np.array(prec.cpu()))
    rec = list(np.array(rec.cpu()))

    acc.extend(f1)
    precs.extend(prec)
    recs.extend(rec)
    # f1 = calculate_f1(y_hat_test, y, None, is_2d=is_2d,
    #     consider_unpairings=consider_unpairings, reduction=False, shift_allowed=shift_allowed, prec_rec=True)

    for k in range(y.size(0)):
        precision, recall, f1_score = evaluate_exact(y_hat_test[k].cpu(), y[k].cpu())
        precision_list.append(precision.item())
        recall_list.append(recall.item())
        f1_list.append(f1_score.item())

    # f1 = list(np.array(f1.cpu()))
    # acc.extend(f1)
    for val, fam in zip(f1, family):
        acc_families[fam][0].append(val)
        acc_families[fam][1].append(y.size(-1))
    lengths.extend([y.size(-1)]*y.size(0))
    families.extend(family)

    if i%100==0:
        L = str(y.size(-1))
        # print('PK', pk_res)
        idx = ['A', 'U', 'U', 'G', 'G', 'C']

        sample_idx = epoch%y_hat_test.size(0)

        if not is_2d:
            n = y_hat_test.size(-1)
            out = torch.zeros((n, n))
            out[range(n), y_hat_test[sample_idx].long().cpu().data] = 1
            out = out * mask[sample_idx]
        else:
            out = y_hat_test[sample_idx, 0] * mask[sample_idx, 0] # [N, N]
            y = y[sample_idx, 0] * mask[sample_idx, 0]
        # tmp = torch.min(torch.argmax(x[sample_idx], 0), 1)[0]
        # seq = [idx[int(i)] for i in tmp]
        with open(log_file, 'a+') as f:
            f.write('PK: %s\n' % str(pk_res))
            f.write('%i\t%f\t[%f]\n' % (x.size(-1), acc[sample_idx-x.size(0)], np.mean(acc)))
        # with open(seq_address % (epoch, 'TEST-%i' % (i), L), 'a+') as f:
        #     f.write(''.join(seq))
        pickle.dump(np.array(torch.argmax(out, -1).cpu()), open(pkl_address % (epoch, 'TEST-%i' % (i), L), 'wb+'))
        # plot(out, y, res_address % (epoch, 'TEST-%i' % (i), L))
