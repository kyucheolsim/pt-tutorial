import sys
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

g_batch_size = 32
g_epochs = 70
g_seq_ext = 30
g_log_steps = 100

# feature size
g_embed_size = 30
g_output_size = 3
g_conv_size = 90
g_kernel_sizes = [3,4,5]

g_dropout_p = 0.2
g_bias = False
g_batchnorm = True
g_learning_rate = 0.0001
g_max_grad_norm = 0
model_path = "seq_cnn_s%s_e%s_h%s_B%d_v0414.pt" % (g_seq_ext, g_embed_size, g_conv_size, g_bias)
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


model = CNN2(g_embed_size, g_conv_size, g_kernel_sizes, g_output_size, g_dropout_p, g_batchnorm, g_bias).to(device)

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
