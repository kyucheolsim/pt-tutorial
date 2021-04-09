import sys
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

g_batch_size = 16
g_epochs = 10
g_seq_ext = 30
g_log_steps = 100

# feature count
#g_vocab_size = 100
# feature size
g_embed_size = 60
g_hidden_size = 100
g_output_size = 3

# num_layers must be larger than or equal to 2, to use dropout in rnn
g_num_layers = 2
g_dropout_p = 0.0
g_batch_first = True
g_bidirectional = False
g_batchnorm = True
g_batchnorm_input = False
g_learning_rate = 0.0001
g_max_grad_norm = 0
g_accuracy = 0.522
#model_path = "seq_model_0.49.pt"
model_path = "seq_model_s%s_e%s_l%s_h%s_b%d_%.3f.pt" % (g_seq_ext, g_embed_size, g_num_layers, g_hidden_size, g_bidirectional, g_accuracy)
print("model_path:", model_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
print("\n* device: ", device)

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
	print("min_lc:", min_lc)
	print("check_list:", check_list)
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
	print("raw count:", len(label_list), len(sidx_list))
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

class LSTMClassifier(nn.Module):
	def __init__(self, embed_size, hidden_size, output_size, num_layers = 3, seq_ext = 30,
		 batch_first = True, bidirectional = True, dropout_p = 0.0, batchnorm = True, batchnorm_input = True):
		super(LSTMClassifier, self).__init__()
		self.embed_size = embed_size
		self.hidden_size = hidden_size
		self.dropout_p = dropout_p
		self.num_layers = num_layers
		self.seq_ext = seq_ext
		if bidirectional:
			self.num_directions = 2
		else:
			self.num_directions = 1

		self.rnn = nn.LSTM(input_size = embed_size,
							 hidden_size = hidden_size,
							 num_layers = num_layers,
							 bias = True,
							 batch_first = batch_first,
							 dropout = dropout_p,
							 bidirectional = bidirectional)

		# fully connected network
		self.linear = nn.Linear(self.num_directions * hidden_size, self.num_directions * hidden_size)
		self.relu = nn.ReLU()
		if dropout_p > 0.0:
			self.dropout = nn.Dropout(p = dropout_p)

		if batchnorm:
			self.batchnorm = nn.BatchNorm1d(self.num_directions * hidden_size)
		else:
			self.batchnorm = None

		if batchnorm_input:
			# C for (N, C, L) or L for (N, L)
			self.batchnorm_input = nn.BatchNorm1d(self.seq_ext)
		else:
			self.batchnorm_input = None

		self.classifier = nn.Linear(self.num_directions * hidden_size, output_size)

	def forward(self, x):
		if self.batchnorm_input:
			embeded = self.batchnorm_input(x)
		else:
			embeded = x
		hidden, cell = self.init_hiddens(x.size(0), device = x.device)
		### output = (batch_size, seq_len, hidden_size * bidirection)
		### hidden = (num_layers * bidirection, batch_size, hidden_size)
		rnn_output, (hidden, cell) = self.rnn(embeded, (hidden, cell))
		#last_hidden = torch.cat([rnn_output[:, -1, : -self.hidden_size], rnn_output[ :, 0, -self.hidden_size : ]], dim = 1)
		last_hidden = torch.cat([h for h in hidden[-self.num_directions : ]], dim = 1)

		if self.batchnorm:
			# don't use batchnorm with dropout
			last_hidden = self.batchnorm(last_hidden)

		if self.dropout_p > 0.0:
			last_hidden = self.dropout(last_hidden)


		# fully connected network
		last_hidden = self.relu(self.linear(last_hidden))

		# batch_size, output_size
		output = self.classifier(last_hidden)
		#return output.view(-1)
		return output
	
	def init_hiddens(self, batch_size, device):
		hidden = torch.zeros(self.num_directions * self.num_layers, batch_size, self.hidden_size)
		cell = torch.zeros(self.num_directions * self.num_layers, batch_size, self.hidden_size)
		return hidden.to(device), cell.to(device)


sidx_list, label_list, feats_list = load_pred_data("./pred_set.txt", seq_ext=g_seq_ext, feat_size=g_embed_size)
print("count(idx, label, feats):", len(sidx_list), len(label_list), len(feats_list))
pred_embedding = torch.tensor(feats_list, requires_grad=False).to(device)
print("* feat_embed_size:", pred_embedding.size(1))
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


model = LSTMClassifier(g_embed_size, g_hidden_size, g_output_size, g_num_layers, g_seq_ext,
		g_batch_first, g_bidirectional, g_dropout_p, g_batchnorm, g_batchnorm_input).to(device)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dic'])
print(checkpoint["best_acc"].item())

num_params = 0
for params in model.parameters():
	num_params += params.view(-1).size(0)
print("\n# of parameters: {}".format(num_params))


model.eval()
with torch.no_grad():
	for i, x in enumerate(pred_loader):
		x = x[0].to(device)
		embed = pred_embedding[x].to(device)
		output = model(embed)
		output_sm = output.softmax(-1).data.cpu().numpy()[0]
		#print(output_sm)
		_, pred = torch.max(output, dim=1)
		print(label_list[i], pred.item(), output_sm[pred.item()], sep="\t")


