import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import classification_report

# precision, recall, f1
def print_report(outputs, labels):
	vlas, inds = torch.max(outputs, dim=1)
	print(classification_report(labels.type(torch.int32).data.cpu().numpy(), inds.data.cpu().numpy()))

# 0 or 1
def get_binary_result(outputs, threshold=0.5):
	outputs[outputs >= threshold] = 1
	outputs[outputs < threshold] = 0
	return outputs

def get_multiclass_correct(class_correct, class_total, preds, labels):
	batch_size = labels.size()[0]
	max_vals, max_inds = torch.max(preds, dim=1)
	#print(max_inds)
	#print(labels)
	#true_false = (preds == labels).squeeze()
	true_false = (max_inds == labels)
	for i in range(batch_size):
		label = labels[i]
		class_correct[label] += true_false[i].item()
		class_total[label] += 1
	return class_correct, class_total

def calc_accuracy(X, Y):
	max_vals, max_indices = torch.max(X, 1)
	acc = (max_indices == Y).sum().cpu().numpy()/max_indices.size()[0]
	return acc

### main ###
# 0, 1, 2
class_count = 3
# output: 3-classes
model = nn.Linear(20, class_count)

# 6 x 20
x = torch.randn(6, 20)
# 1 x 6
y = torch.tensor([1, 0, 1, 0, 2, 0])

loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-1)

### train ###
model.train()
for epoch in range(10):
	train_correct = list(0. for i in range(class_count))
	train_total = list(0. for i in range(class_count))
	optimizer.zero_grad()
	output_raw = model(x)
	loss = loss_func(output_raw, y)
	acc = calc_accuracy(output_raw, y)
	loss.backward()
	optimizer.step()
	print("epoch: %s, loss: %.3f, acc: %.3f" % (epoch, loss.item(), acc.item()))
	train_correct, train_total  = get_multiclass_correct(train_correct, train_total, output_raw, y)
	#print(train_correct)
	#print(train_total)
	print(np.array(train_correct)/np.array(train_total))


### validation ###
#with torch.no_grad():
model.eval()
preds = model(x)
valid_correct = list(0. for i in range(class_count))
valid_total = list(0. for i in range(class_count))
valid_correct, valid_total  = get_multiclass_correct(valid_correct, valid_total, preds, y)
print(np.array(valid_correct)/np.array(valid_total))
print_report(preds, y)
