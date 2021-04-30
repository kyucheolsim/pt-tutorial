import sys
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

g_batch_size = 32
g_epochs = 70
g_seq_ext = 30
g_log_steps = 100

# feature count
#g_vocab_size = 100
# feature size
g_embed_size = 30
g_hidden_size = 90
g_output_size = 3

# num_layers must be larger than or equal to 2, to use dropout in rnn
g_num_layers = 2
g_dropout_p = 0.5
g_bias = False
g_batch_first = True
g_bidirectional = True
g_batchnorm_input = False
g_batchnorm = True
g_learning_rate = 0.0001
g_max_grad_norm = 0
model_path = "seq_model_s%s_e%s_l%s_h%s_b%d_B%d_v0414.pt" % (g_seq_ext, g_embed_size, g_num_layers, g_hidden_size, g_bidirectional, g_bias)
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


# sequence data sets: label features _EOS_ label features
def load_seq_data(data_path, seq_ext=10, feat_size=0, eos="_EOS_"):
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
			label = int(tokens[0])
			if feat_size > 0:
				feats = [float(f) for f in tokens[1:feat_size+1]]
			else:
				feats = [float(f) for f in tokens[1:]]
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
	print("raw label count:", len(label_list))
	label_list, sidx_list = get_equal_set(label_list, sidx_list)
	print("equ label count:", len(label_list))
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

### train set
sidx_list, label_list, feats_list = load_seq_data("./train_set.txt", seq_ext=g_seq_ext, feat_size=g_embed_size)
print("count(idx, label, feats):", len(sidx_list), len(label_list), len(feats_list))
train_embedding = torch.tensor(feats_list, requires_grad=False).to(device)
print("* train_embed_size:", train_embedding.size(1))
g_embed_size = len(feats_list[0])
print("train_embed_size:", g_embed_size)

train_x = []
for i in sidx_list:
	seq_data = get_seq_list(i, seq_ext=g_seq_ext, tensor=False, reverse=True)
	train_x.append(seq_data)
	#print(i, seq_data)
train_x = torch.tensor(train_x)
train_y = torch.tensor(label_list)

print("* train_size:", train_x.size())
#print("idx_0:", train_x[torch.tensor(0)])

### valid set
sidx_list, label_list, feats_list = load_seq_data("./valid_set.txt", seq_ext=g_seq_ext, feat_size=g_embed_size)
valid_embedding = torch.tensor(feats_list, requires_grad=False).to(device)
print("* valid_embed_size:", valid_embedding.size(1))

valid_x = []
for i in sidx_list:
	seq_data = get_seq_list(i, seq_ext=g_seq_ext, tensor=False, reverse=True)
	valid_x.append(seq_data)
valid_x = torch.tensor(valid_x)
valid_y = torch.tensor(label_list)
print("* valid_size:", valid_x.size())


train_set = TensorDataset(train_x, train_y)
train_loader = DataLoader(train_set, batch_size=g_batch_size, shuffle=True)

valid_set = TensorDataset(valid_x, valid_y)
valid_loader = DataLoader(valid_set, batch_size=g_batch_size, shuffle=False)


class LSTMClassifier(nn.Module):
	def __init__(self, embed_size, hidden_size, output_size, num_layers = 3, seq_ext = 30,
		 batch_first = True, bidirectional = True, dropout_p = 0.0, batchnorm = True, batchnorm_input = True, bias=True):
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
							 bias = bias,
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
			embedded = self.batchnorm_input(x)
		else:
			embedded = x
		hidden, cell = self.init_hiddens(x.size(0), device = x.device)
		### output = (batch_size, seq_len, hidden_size * bidirection)
		### hidden = (num_layers * bidirection, batch_size, hidden_size)
		rnn_output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
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


model = LSTMClassifier(g_embed_size, g_hidden_size, g_output_size, g_num_layers, g_seq_ext,
		g_batch_first, g_bidirectional, g_dropout_p, g_batchnorm, g_batchnorm_input, g_bias).to(device)

#print(model)
num_params = 0
for params in model.parameters():
	num_params += params.view(-1).size(0)
print("\n# of parameters: {}".format(num_params))

loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=g_learning_rate)

best_acc = 0.0
best_epo = 0
for e in range(g_epochs):
	model.train()
	train_size = len(train_loader.dataset)
	train_acc = 0.0
	for i, (x, y) in enumerate(train_loader):
		x, label = x.to(device), y.to(device)
		embed = train_embedding[x].to(device)
		optimizer.zero_grad()
		output = model(embed)
		loss = loss_func(output, label)
		loss.backward()
		if g_max_grad_norm > 0:
			torch.nn.utils.clip_grad_norm_(model.parameters(), g_max_grad_norm)
		optimizer.step()
		train_acc += calc_accuracy(output, label)
		if i % g_log_steps == 0:
			print("- train e:{} ({:05.2f}%), loss: {:.5f}, accurcy: {:.5f}".format(e + 1, (100.* i * train_loader.batch_size)/train_size, loss.item(), train_acc/(i + 1)))

	mc_correct, mc_total = init_mc_correct(g_output_size)
	model.eval()
	with torch.no_grad():
		valid_size = len(valid_loader.dataset)
		valid_correct = 0.0
		for x, y in valid_loader:
			x, label = x.to(device), y.to(device)
			embed = valid_embedding[x].to(device)
			output = model(embed)
			valid_correct += calc_accuracy(output, label, correct_count=True)
			mc_correct, mc_total = get_mc_correct(mc_correct, mc_total, output, label, predict=True)
		valid_acc = valid_correct/valid_size
		print("* valid_acc: {:.5f}".format(valid_acc))
		print("-", list(map(lambda x:round(x, 3), np.array(mc_correct)/np.array(mc_total))))
		if valid_acc > best_acc:
			best_val = list(map(lambda x:round(x, 3), np.array(mc_correct)/np.array(mc_total)))
			print(list(zip(mc_correct, mc_total)))
			best_epo = e
			best_acc = valid_acc
			torch.save({"best_acc": best_acc, "model_state_dic": model.state_dict()}, model_path)
		print("* best_epo: {}, best_acc: {:.5f}".format(best_epo, best_acc))
		print("* best_val:", best_val)
