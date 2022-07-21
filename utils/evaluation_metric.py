import numpy as np
from sklearn.metrics import f1_score


def accuracy(input, target):
    assert len(input.shape) == 1
    return sum(input==target)/input.shape[0]


def averaged_f1_score(input, target):
	N, label_size = input.shape
	f1s = []
	for i in range(label_size):
		f1 = f1_score(input[:, i], target[:, i])
		f1s.append(f1)
	return np.mean(f1s), f1s


def EXPR_metric(x, y):
    # x: predict; y: target
    if not len(x.shape) == 1:
        if x.shape[1] == 1:
            x = x.reshape(-1)
        else:
            x = np.argmax(x, axis=-1)
    if not len(y.shape) == 1:
        if y.shape[1] == 1:
            y = y.reshape(-1)
        else:
            y = np.argmax(y, axis=-1)

    f1 = f1_score(x, y, average= 'macro')
    acc = accuracy(x, y)
    return f1, acc, 0.67*f1 + 0.33*acc


def AU_metric(x, y):
	f1_av,_  = averaged_f1_score(x, y)
	x = x.reshape(-1)
	y = y.reshape(-1)
	acc_av  = accuracy(x, y)
	return f1_av, acc_av, 0.5*f1_av + 0.5*acc_av