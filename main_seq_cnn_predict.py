import sys
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader
#from tqdm import tqdm

PARAM = None
parser = argparse.ArgumentParser()
parser.add_argument(
'--action', type=str,
default='', help=''
)
PARAM = parser.parse_args()

if PARAM.action == "long":
	LONG_MODEL = True
else:
	LONG_MODEL = False

g_batch_size = 32
g_epochs = 100
g_log_steps = 100

### input sequence length
if LONG_MODEL:
	g_seq_ext = 40 # long
else:
	g_seq_ext = 30 # short

### feature size
if LONG_MODEL:
	g_embed_size = 45 # long (embed + sise 5)
else:
	g_embed_size = 35 # short (embed + sise 5)
g_conv_size = 60
g_kernel_sizes = [3,4,5]
g_output_size = 3

# num_layers must be larger than or equal to 2, to use dropout in rnn
g_dropout_p = 0.1
g_batchnorm = True
g_bias = True
g_learning_rate = 0.0001
g_max_grad_norm = 0

if LONG_MODEL:
	model_path = "./model/pred5/seq_cnn5_s40_e45_h60_B1_v0419_0.6198.pt"
	pred_path = "./pred_set/pred_set_long.txt"
else:
	model_path = "./model/pred3/seq_cnn3_s30_e35_h60_B1_v0419_0.5516.pt"
	pred_path = "./pred_set/pred_set_short.txt"
print("# model_path:", model_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
print("# device: ", device)

def get_min_label_count(label_list):
	min_lc = 0
	label_dic = {}
	for label in label_list:
		label_dic[label] = label_dic.get(label, 0) + 1
	for label, count in sorted(label_dic.items(), key=lambda x:x[1]):
		#print(label, count)
		if min_lc == 0:
			# first label count
			min_lc = count
		else:
			label_dic[label] = min_lc
	return min_lc, label_dic

def get_equal_set(label_list, sidx_list):
	min_lc, label_dic = get_min_label_count(label_list)
	check_list = [min_lc] * len(label_dic)
	print("# min_lc:", min_lc)
	print("# check_list:", check_list)
	li_pair = list(zip(label_list, sidx_list))
	random.seed(23)
	random.shuffle(li_pair)
	equal_list = []
	for i, li in enumerate(li_pair):
		if check_list[li[0]] > 0:
			check_list[li[0]] -= 1
			equal_list.append(li)
		if sum(check_list) <= 0:
			break
	label_list, sidx_list = zip(*equal_list)
	return label_list, sidx_list

# sequence data sets: code info features _EOS_ code info features
def load_pred_data(data_path, seq_ext=10, feat_size=0, eos="_EOS_"):
	if seq_ext > 1:
		rm_from_idx = -(seq_ext-1)
	else:
		rm_from_idx = None 
	last_idx = 0
	# seq: start_idx == label_list
	sidx_list, label_list, feats_list = [], [], []
	with open(data_path, "r") as fin:
		idx = 0
		for line in fin:
			line = line.strip()
			if not line or line[0] == "#": continue
			if line == eos:
				last_idx = idx
				#print("_EOS_", idx)
				#print("before", len(sidx_list), sidx_list[-1])
				if seq_ext > 1:
					del(sidx_list[rm_from_idx:])
					del(label_list[rm_from_idx:])
				#print("after", len(sidx_list), sidx_list[-1])
				continue
			tokens = line.split("\t")
			code = tokens[0]
			info = tokens[1]
			label = "%s\t%s" % (code, info)
			if feat_size > 0:
				feats = [float(f) for f in tokens[2:feat_size+2]]
			else:
				feats = [float(f) for f in tokens[2:]]
			label_list.append(label)
			feats_list.append(feats)
			sidx_list.append(idx)
			idx += 1

		if last_idx != idx:
			# handle last data block
			if seq_ext > 1:
				del(sidx_list[rm_from_idx:])
				del(label_list[rm_from_idx:])
		#print("idx:", last_idx, idx, len(sidx_list))
	print("# raw count:", len(label_list), len(sidx_list))
	#label_list, sidx_list = get_equal_set(label_list, sidx_list)
	#print("equ count:", len(label_list), len(sidx_list))
	return sidx_list, label_list, feats_list

def get_seq_list(start_idx, seq_ext=10, tensor=False, reverse=True):
	if tensor:
		if reverse:
			return torch.tensor(range(start_idx+seq_ext-1, start_idx-1, -1))
		else:
			return torch.tensor(range(start_idx, start_idx+seq_ext, 1))
	else:
		if reverse:
			return range(start_idx+seq_ext-1, start_idx-1, -1)
		else:
			return range(start_idx, start_idx+seq_ext, 1)

def calc_accuracy(X, Y, correct_count = False):
	_, pred = torch.max(X, 1)
	if correct_count:
		accuracy = (pred.view(Y.size()) == Y).sum()
	else:
		accuracy = (pred.view(Y.size()) == Y).sum()/torch.tensor(Y.size()[0], dtype=torch.float64)
	return accuracy

def init_mc_correct(class_count):
	correct = list(0. for i in range(class_count))
	total = list(0. for i in range(class_count))
	return correct, total

def get_mc_correct(class_correct, class_total, preds, labels, predict=False):
	batch_size = labels.size()[0]
	max_vals, max_inds = torch.max(preds, dim=1)
	#true_false = (preds == labels).squeeze()
	true_false = (max_inds == labels)
	for i in range(batch_size):
		if predict:
			label = max_inds[i]
		else:
			label = labels[i]
		class_correct[label] += true_false[i].item()
		class_total[label] += 1
	return class_correct, class_total


class CNN2(nn.Module):
	def __init__(self, embed_size, conv_size, kernel_sizes, output_size, dropout_p, batchnorm, bias):
		super(CNN2, self).__init__()
		self.embed_size = embed_size
		self.conv_size = conv_size
		self.dropout_p = dropout_p

		self.convs = nn.ModuleList([
			nn.Conv2d(in_channels = 1,
			out_channels = conv_size,
			kernel_size = (ks, embed_size),
			stride = 1,
			bias = bias)
			for ks in kernel_sizes])

		# fully connected network
		self.linear = nn.Linear(len(kernel_sizes) * conv_size, conv_size)

		if dropout_p > 0.0:
			self.dropout = nn.Dropout(p = dropout_p)

		if batchnorm:
			self.batchnorm = nn.BatchNorm1d(len(kernel_sizes) * conv_size)
		else:
			self.batchnorm = None

		self.classifier = nn.Linear(conv_size, output_size)

	def forward(self, x):
		#x = [batch_size, seq_len]
		#embedded = self.embedding(x)
		#embedded = [batch_size, seq_len, emb_size]
		embedded = x
		#embedded = [batch_size, 1, seq_len, emb_size]
		embedded = embedded.unsqueeze(1)
		#conved_n = [batch_size, conv_size, seq_len - kernel_sizes[n] + 1]
		conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
		#pooled_n = [batch_size, conv_size]
		pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
		#cated = [batch_size, conv_size * len(kernel_sizes)]
		cated = torch.cat(pooled, dim = 1)

		if self.batchnorm:
			# don't use batchnorm with dropout
			cated = self.batchnorm(cated)

		if self.dropout_p > 0.0:
			cated = self.dropout(cated)

		# fully connected network
		lineared = F.relu(self.linear(cated))
		# batch_size, output_size
		output = self.classifier(lineared)
		#return output.view(-1)
		return output


sidx_list, label_list, feats_list = load_pred_data(pred_path, seq_ext=g_seq_ext, feat_size=g_embed_size)
print("# count(idx, label, feats):", len(sidx_list), len(label_list), len(feats_list))
pred_embedding = torch.tensor(feats_list, requires_grad=False).to(device)
print("# feat_embed_size:", pred_embedding.size(1))
g_embed_size = len(feats_list[0])

pred_x = []
for i in sidx_list:
	seq_data = get_seq_list(i, seq_ext=g_seq_ext, tensor=False, reverse=True)
	pred_x.append(seq_data)
	#print(i, seq_data)
pred_x = torch.tensor(pred_x)
#print(pred_x)

pred_set = TensorDataset(pred_x)
pred_loader = DataLoader(pred_set, batch_size=1, shuffle=False)
data_size = len(pred_loader.dataset)


model = CNN2(g_embed_size, g_conv_size, g_kernel_sizes, g_output_size, g_dropout_p, g_batchnorm, g_bias).to(device)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dic'])
print("# best_acc:", checkpoint["best_acc"].item())

num_params = 0
for params in model.parameters():
	num_params += params.view(-1).size(0)
print("# of parameters: {}".format(num_params))


model.eval()
with torch.no_grad():
	for i, x in enumerate(pred_loader):
		x = x[0].to(device)
		embed = pred_embedding[x].to(device)
		output = model(embed)
		output_sm = output.softmax(-1).data.cpu().numpy()[0]
		#print(output_sm)
		_, pred = torch.max(output, dim=1)
		print(label_list[i], pred.item(), "%.5f"%(output_sm[pred.item()]), sep="\t")


