"""Parsing the parameters."""
import argparse

import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def parameter_parser():
    parser = argparse.ArgumentParser(description="Run GETNext.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help='Random seed')
    parser.add_argument('--device',
                        type=str,
                        default=device,
                        help='')
    parser.add_argument('--data-adj-mtx',
                        type=str,
                        default='dataset/NYC/graph_A.csv',
                        help='Graph adjacent path')
    parser.add_argument('--data-node-feats',
                        type=str,
                        default='dataset/NYC/graph_X.csv',
                        help='Graph node features path')
    parser.add_argument('--data-train',
                        type=str,
                        default='dataset/NYC/NYC_train_with_categories.csv',
                        help='Training data path')
    parser.add_argument('--data-val',
                        type=str,
                        default='dataset/NYC/NYC_val_with_categories.csv',
                        help='Validation data path')
    parser.add_argument('--short-traj-thres',
                        type=int,
                        default=4,
                        help='Remove over-short trajectory')
    parser.add_argument('--time-units',
                        type=int,
                        default=48,
                        help='Time unit is 0.5 hour, 24/0.5=48')
    parser.add_argument('--time-feature',
                        type=str,
                        default='norm_in_day_time',
                        help='The name of time feature in the data')

    # Model hyper-parameters
    parser.add_argument('--poi-embed-dim',
                        type=int,
                        default=128,
                        help='POI embedding dimensions')
    parser.add_argument('--user-embed-dim',
                        type=int,
                        default=128,
                        help='User embedding dimensions')
    parser.add_argument('--gating-hidden-dim',
                        type=int,
                        default=128,
                        help='gating-hidden-dim')
    parser.add_argument('--time-embed-dim',
                        type=int,
                        default=32,
                        help='Time embedding dimensions')
    parser.add_argument('--week-embed-dim',
                        type=int,
                        default=32,
                        help='week embedding dimensions')
    parser.add_argument('--geo-embed-dim',
                        type=int,
                        default=32,
                        help='geo embedding dimensions')

    parser.add_argument('--cat-embed-dim',
                        type=int,
                        default=32,
                        help='Category embedding dimensions')
    parser.add_argument('--cat2-embed-dim',
                        type=int,
                        default=32,
                        help='Time embedding dimensions')


    parser.add_argument('--gcn-dropout',
                        type=float,
                        default=0.35,
                        help='Dropout rate for gcn')
    parser.add_argument('--gcn-nhid',
                        type=list,
                        default=[32, 64],
                        help='List of hidden dims for gcn layers')
    parser.add_argument('--transformer-nhid',
                        type=int,
                        default=1024,
                        help='Hid dim in TransformerEncoder')
    parser.add_argument('--transformer-nlayers',
                        type=int,
                        default=4,
                        help='Num of TransformerEncoderLayer')
    parser.add_argument('--transformer-nhead',
                        type=int,
                        default=4,
                        help='Num of heads in multiheadattention')
    parser.add_argument('--transformer-dropout',
                        type=float,
                        default=0.2,
                        help='Dropout rate for transformer')

    parser.add_argument('--time-loss-weight',
                        type=int,
                        default=10,
                        help='Scale factor for the time loss term')

    # Training hyper-parameters
    parser.add_argument('--batch',
                        type=int,
                        default=20,
                        help='Batch size.')
    parser.add_argument('--epochs',
                        type=int,
                        default=200,
                        help='Number of epochs to train.')
    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--lr-scheduler-factor',
                        type=float,
                        default=0.1,
                        help='Learning rate scheduler factor')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=5e-4,
                        help='Weight decay (L2 loss on parameters).')

    # Experiment config
    parser.add_argument('--save-weights',
                        action='store_true',
                        default=True,
                        help='whether save the model')
    parser.add_argument('--save-embeds',
                        action='store_true',
                        default=False,
                        help='whether save the embeddings')
    parser.add_argument('--workers',
                        type=int,
                        default=0,
                        help='Num of workers for dataloader.')
    parser.add_argument('--project',
                        default='runs/train',
                        help='save to project/name')
    parser.add_argument('--name',
                        default='exp',
                        help='save to project/name')
    parser.add_argument('--exist-ok',
                        action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--no-cuda',
                        action='store_true',
                        default=False, help='Disables CUDA training.')
    parser.add_argument('--mode',
                        type=str,
                        default='client',
                        help='python console use only')
    parser.add_argument('--port',
                        type=int,
                        default=64973,
                        help='python console use only')
    parser.add_argument('--fuse_way', type=str, default='manual',
                      choices=['adaptive', 'concat', 'manual'],
                      help='POI特征融合方式: adaptive(自适应权重), concat(拼接对齐), manual(手动加权)')
    parser.add_argument('--similar_user_k', type=int, default=5,
                      help='相似用户数量')
    return parser.parse_args()
