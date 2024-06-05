from torch import nn
import argparse
import random
import numpy as np
import torch
import os


class CommendArg(nn.Module):
    def get_parser(self):
        parser = argparse.ArgumentParser(description='Parser For Arguments',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        parser.add_argument('--name', default='lte', help='Set run name for saving/restoring models')
        parser.add_argument('--dataset', dest='dataset', default='FB15k-237', help='Dataset to use, default: FB15k-237')
        parser.add_argument('--score_func', dest='score_func', default='transe', help='Score Function for Link prediction')
        parser.add_argument('--opn', dest='opn', default='mult', help='Composition Operation to be used in CompGCN')

        parser.add_argument('--batch', dest='batch_size', default=256, type=int, help='Batch size')
        parser.add_argument('--device', default='cpu', help='choose CPU or GPU')
        parser.add_argument('--epoch', dest='max_epochs', type=int, default=500, help='Number of epochs')
        parser.add_argument('--l2', type=float, default=0.0, help='L2 Regularization for Optimizer')
        parser.add_argument('--lr', type=float, default=0.001, help='Starting Learning Rate')
        parser.add_argument('--lbl_smooth', dest='lbl_smooth', type=float, default=0.1, help='Label Smoothing')
        parser.add_argument('--num_workers', type=int, default=1, help='Number of processes to construct batches')
        parser.add_argument('--seed', dest='seed', default=12345, type=int, help='Seed for randomization')

        parser.add_argument('--restore', dest='restore', action='store_true', help='Restore from the previously saved model')
        parser.add_argument('--bias', dest='bias', action='store_true', help='Whether to use bias in the model')
        parser.add_argument('--init_dim', dest='init_dim', default=200, type=int, help='Initial dimension size for entities and relations')
        parser.add_argument('--gcn_dim', dest='gcn_dim', default=200, type=int, help='Number of hidden units in GCN')
        parser.add_argument('--n_layer', dest='n_layer', default=0, type=int, help='Number of GCN Layers to use')

        parser.add_argument('--gamma', dest='gamma', default=9.0, type=float, help='TransE: Gamma to use')

        parser.add_argument('--noltr', action='store_true', default=False, help='no use of linear transformations for relation embeddings')

        parser.add_argument('--encoder', dest='encoder', default='compgcn', type=str, help='which encoder to use')
        parser.add_argument('--hid_drop', dest='hid_drop', default=0.2, type=float, help='Dropout after GCN')

        parser.add_argument('--x_ops', dest='x_ops', default="p")
        parser.add_argument('--r_ops', dest='r_ops', default="")

        parser.add_argument('--conve_hid_drop', dest='conve_hid_drop', default=0.3, type=float,
                        help='ConvE: Hidden dropout')
        parser.add_argument('--feat_drop', dest='feat_drop',
                            default=0.2, type=float, help='ConvE: Feature Dropout')
        parser.add_argument('--input_drop', dest='input_drop', default=0.2,
                            type=float, help='ConvE: Stacked Input Dropout')
        parser.add_argument('--k_w', dest='k_w', default=20,
                            type=int, help='ConvE: k_w')
        parser.add_argument('--k_h', dest='k_h', default=10,
                            type=int, help='ConvE: k_h')
        parser.add_argument('--num_filt', dest='num_filt', default=200, type=int,
                            help='ConvE: Number of filters in convolution')
        parser.add_argument('--ker_sz', dest='ker_sz', default=7,
                            type=int, help='ConvE: Kernel size to use')
        parser.add_argument('--embed_dim', dest='embed_dim', default=100, type=int,
                        help='Embedding dimension to give as input to score function')
        

        args = parser.parse_args()
        if not args.restore:
            args.name = args.encoder.lower() + '-' + args.score_func.lower() + '-' + args.opn + args.name

        # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        # torch.cuda.manual_seed(args.seed)
        torch.manual_seed(args.seed)

        return args