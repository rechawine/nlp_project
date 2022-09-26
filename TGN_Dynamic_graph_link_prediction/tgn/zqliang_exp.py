import torch
import numpy as np
from torch.nn.parameter import Parameter
import torch.nn as nn
from torch.autograd import Variable 
import argparse
from utils.data_processing import get_data, compute_time_statistics

### Argument and global variables
parser = argparse.ArgumentParser('TGN self-supervised training')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='wikipedia')
# parser.add_argument('--bs', type=int, default=200, help='Batch_size')
parser.add_argument('--bs', type=int, default=64, help='Batch_size')
parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=50, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to '
                                                                  'backprop')
parser.add_argument('--use_memory', action='store_true',
                    help='Whether to augment the model with a node memory')
parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
  "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
parser.add_argument('--message_function', type=str, default="identity", choices=[
  "mlp", "identity"], help='Type of message function')
parser.add_argument('--memory_updater', type=str, default="gru", choices=[
  "gru", "rnn"], help='Type of memory updater')
parser.add_argument('--aggregator', type=str, default="last", help='Type of message '
                                                                        'aggregator')
parser.add_argument('--memory_update_at_end', action='store_true',
                    help='Whether to update memory at the end or at the start of the batch')
parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
parser.add_argument('--memory_dim', type=int, default=172, help='Dimensions of the memory for '
                                                                'each user')
parser.add_argument('--different_new_nodes', action='store_true',
                    help='Whether to use disjoint set of new nodes for train and val')
parser.add_argument('--uniform', action='store_true',
                    help='take uniform sampling from temporal neighbors')
parser.add_argument('--randomize_features', action='store_true',
                    help='Whether to randomize node features')
parser.add_argument('--use_destination_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the destination node as part of the message')
parser.add_argument('--use_source_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the source node as part of the message')
parser.add_argument('--dyrep', action='store_true',
                    help='Whether to run the dyrep model')
# additional arguments
parser.add_argument('--val_ratio', type=float, default=0.15, help='Ratio of the validation data.')
parser.add_argument('--test_ratio', type=float, default=0.15, help="Ratio of the test data.")


try:
  args = parser.parse_args()
except:
  parser.print_help()
  sys.exit(0)

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
USE_MEMORY = args.use_memory
MESSAGE_DIM = args.message_dim
MEMORY_DIM = args.memory_dim

# class Model(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.m = Parameter(torch.randn(9,2,10))
#     def forward(self, a):
#         out = a * self.m
#         return out

# if __name__ == "__main__":
#     input_Data = torch.randn(9,2,10).cuda()

#     model = Model().cuda()
#     optim = torch.optim.SGD(model.parameters(), 1e-3)
#     model.train()

#     for i in range(100):
#         optim.zero_grad()
#         out = model(input_Data).sum()
#         out.backward()
#         optim.step()
#         print(f"iter is {i}", f"loss is {out.item()}")
'''
torch.nn.Embedding(
num_embeddings, – 词典的大小尺寸，比如总共出现5000个词，那就输入5000。此时index为（0-4999）
embedding_dim,– 嵌入向量的维度，即用多少维来表示一个符号。
padding_idx=None,– 填充id，比如，输入长度为100，但是每次的句子长度并不一样，后面就需要用统一的数字填充，而这里就是指定这个数字，这样，网络在遇到填充id时，就不会计算其与其它符号的相关性。（初始化为0）
max_norm=None, – 最大范数，如果嵌入向量的范数超过了这个界限，就要进行再归一化。
norm_type=2.0, – 指定利用什么范数计算，并用于对比max_norm，默认为2范数。
scale_grad_by_freq=False, 根据单词在mini-batch中出现的频率，对梯度进行放缩。默认为False.
sparse=False, – 若为True,则与权重矩阵相关的梯度转变为稀疏张量。
_weight=None)
输出：
[规整后的句子长度，样本个数（batch_size）,词向量维度]
'''
# num = [23122, 55, 621.8, 11023] # (23122+11023) / 55 = 621.8

# length = len(num)
# print(length)
# embeddings = torch.randn(1, 512,10)
# print(embeddings(num))

# print(embeddings(num).size())

'''
The dataset has 157474 interactions, involving 8227 different nodes
The training dataset has 80161 interactions, involving 5400 different nodes
The validation dataset has 23621 interactions, involving 2878 different nodes
The test dataset has 23621 interactions, involving 3181 different nodes
The new node validation dataset has 10054 interactions, involving 1936 different nodes
The new node test dataset has 9755 interactions, involving 2230 different nodes
822 nodes were used for the inductive testing, i.e. are never seen during training
'''
### Extract data for training, validation and testing
node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, \
new_node_test_data = get_data(DATA, args.val_ratio, args.test_ratio,
                              different_new_nodes_between_val_and_test=args.different_new_nodes,
                              randomize_features=args.randomize_features)

print(node_features)
print(edge_features, type(edge_features))
print(full_data[:20].sources,len(full_data.sources), full_data[:20].destinations, len(full_data.destinations))
