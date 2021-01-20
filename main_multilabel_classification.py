import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import classification_report

# batch_size x pos_values
def get_positive_values(outputs):
	indexes = []
	values = []
	for output in outputs:
		index = (output > 0.0).nonzero()
		indexes.append(index.data.numpy().flatten())
		values.append(output[index].data.numpy().flatten())
	return indexes, values

# batch_size x pos_values: np.array => [ [1,2,3], [ ], [1,3,5] ]
def print_result(pos_indexes, pos_values):
	for i, pos_idx in enumerate(pos_indexes):
		if pos_idx.any():
			for j, idx in enumerate(pos_idx):
				print("batch: %s, index: %s => %s" % (i, idx, pos_values[i][j]))
		else:
			print("batch: %s, index: N => N" % (i))

# batch_size x class_count
def get_multilabel_accuracy(outputs, labels, threshold=0.5):
	labels = labels.type(torch.int32)
	batch_size, class_count = outputs.size()
	total = batch_size * class_count
	outputs[outputs >= threshold] = 1
	outputs[outputs < threshold] = 0
	accuracy_sum = (outputs == labels).sum()
	accuracy = accuracy_sum.type(torch.float32)/total
	#print(accuracy.item(), accuracy_sum.item(), total)
	return accuracy

# precision, recall, f1
def print_classification_report(outputs, labels, threshold=0.5):
	outputs[outputs >= threshold] = 1
	outputs[outputs < threshold] = 0
	print(classification_report(labels.type(torch.int32).data.cpu().numpy(), outputs.data.cpu().numpy()))

# 0 or 1
def get_binary_result(outputs, threshold=0.5):
	outputs[outputs >= threshold] = 1
	outputs[outputs < threshold] = 0
	return outputs

# preds, labels: 0 or 1 (onehot encoding)
def get_multilabel_class_accuracy(preds, labels):
	batch_size, class_count = labels.size()
	class_correct = list(0. for i in range(class_count))
	class_total = list(0. for i in range(class_count))
	#true_false = (preds == labels).squeeze()
	true_false = (preds == labels)
	for i in range(batch_size):
		print(true_false[i])
		class_correct += true_false[i].data.cpu().numpy()
		class_total += np.array(1)
	class_accuracy = class_correct/class_total
	print(class_correct)
	print(class_total)
	print(class_accuracy)
	return class_accuracy

# preds, labels: 0 or 1 (onehot encoding), accumulated
def get_multilabel_class_correct(class_correct, class_total, preds, labels):
	batch_size, class_count = labels.size()
	#true_false = (preds == labels).squeeze()
	true_false = (preds == labels)
	for i in range(batch_size):
		print(true_false[i])
		class_correct += true_false[i].data.cpu().numpy()
		class_total += np.array(1)
	print(class_correct)
	print(class_total)
	return class_correct, class_total


### main ###
model = nn.Linear(20, 5)

# batch: 2 x 20
x = torch.randn(2, 20)
#x = torch.randn(1, 20)

# 2 x 5 (onehot encoding)
y = torch.tensor([[1., 0., 1., 0., 0.], [1., 0., 0., 0., 1.]])
#y = torch.tensor([[1., 0., 1., 0., 0.]])

# outputs before sigmoid
loss_func_raw = nn.BCEWithLogitsLoss()

# outputs after sigmoid
loss_func_sigmoid = nn.BCELoss()

optimizer = optim.SGD(model.parameters(), lr=1e-1)

### train ###
model.train()
for epoch in range(5):
	optimizer.zero_grad()
	output_raw = model(x)
	output_sm = torch.sigmoid(output_raw)
	#print(output_raw)
	#print(output_sm)

	#print_classification_report(output_sm, y, threshold=0.5)

	# batch_size x class_count
	accuracy = get_multilabel_accuracy(output_sm, y, threshold=0.5)

	# batch_size x pos_values
	#pos_indexes, pos_values = get_positive_values(output_raw)
	#print_result(pos_indexes, pos_values)

	loss = loss_func_raw(output_raw, y)
	#loss = loss_func_sigmoid(output_sm, y)

	loss.backward()
	optimizer.step()
	print("epoch: %s, loss: %.3f, acc: %.3f" % (epoch, loss.item(), accuracy.item()))


### validation ###
#with torch.no_grad():
model.eval()
preds = model(x)
#print(preds)
preds = get_binary_result(preds, threshold=0.0)
#print(preds)

class_count = y.size()[1]
class_correct = list(0. for i in range(class_count))
class_total = list(0. for i in range(class_count))
# accumulated
class_correct, class_total  = get_multilabel_class_correct(class_correct, class_total, preds, y)
#print(class_correct/class_total)

# not accumulated
class_accuracy = get_multilabel_class_accuracy(preds, y)

