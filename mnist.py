import os
from urllib import request
import gzip
import numpy as np

import framework as lib


class MNIST:
    FILES = [
        "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"
    ]

    URL = "http://yann.lecun.com/exdb/mnist/"

    @staticmethod
    def gzload(file, offset):
        with gzip.open(file, 'rb') as f:
            return np.frombuffer(f.read(), np.uint8, offset=offset)

    def __init__(self, set, cache="./cache"):
        os.makedirs(cache, exist_ok=True)

        for name in self.FILES:
            path = os.path.join(cache, name)
            if not os.path.isfile(path):
                print("Downloading " + name)
                request.urlretrieve(self.URL + name, path)

        if set=="test":
            f_offset = 2
        elif set=="train":
            f_offset = 0
        else:
            assert False, "Invalid set: "+set

        self.images = self.gzload(os.path.join(cache, self.FILES[f_offset]), 16).reshape(-1,28*28).astype(np.float)/255.0
        self.labels = self.gzload(os.path.join(cache, self.FILES[f_offset+1]), 8)

    def __len__(self):
        return self.images.shape[0]


class SoftmaxCrossEntropyLoss:
    @staticmethod
    def _softmax(input):
        input = input - np.max(input, axis=-1, keepdims=True)
        e = np.exp(input)
        return e / e.sum(axis=-1, keepdims=True)

    def forward(self, net_output, targets):
        self.saved_variables = {
            "out": net_output,
            "targets": targets
        }

        out = self._softmax(net_output)
        return np.mean(-np.log(out[range(net_output.shape[0]), targets]))

    def backward(self):
        net_output = self.saved_variables["out"]
        targets = self.saved_variables["targets"]

        batch_size = net_output.shape[0]
        grad = self._softmax(net_output)
        grad[range(batch_size), targets] -= 1

        self.saved_variables = None
        return grad / batch_size

train_validation_set = MNIST("train")
test_set = MNIST("test")

n_train = int(0.7 * len(train_validation_set))
print("MNIST:")
print("   Train set size:", n_train)
print("   Validation set size:", len(train_validation_set) - n_train)
print("   Test set size", len(test_set))

np.random.seed(0xDEADBEEF)
batch_size = 64

loss = SoftmaxCrossEntropyLoss()
learning_rate = 0.03

model = lib.Sequential([
    lib.Linear(28*28, 20),
    lib.Tanh(),
    lib.Linear(20, 10)
])

indices = np.random.permutation(len(train_validation_set))

train_indices = indices[0:n_train]
validation_indices  = indices[n_train: ]

def verify(images, targets):
    num_ok_ = 0
    total_num_ = 0

    for i in range(0, len(images), batch_size):
        images_ = images[i: i+batch_size]
        targets_ = targets[i: i+batch_size]
        model_output = model.forward(images_)
        max_elem = np.argmax(model_output, axis=1)

        num_ok_ += np.sum(max_elem == targets_)
        total_num_ += len(max_elem)
        
    return num_ok_, total_num_

def validate():    
    accu = 0.0
    count = 0
    accu, count = verify(train_validation_set.images[validation_indices], train_validation_set.labels[validation_indices])

    return accu/count * 100.0

def test():
    accu = 0.0
    count = 0
    accu, count = verify(test_set.images, test_set.labels)

    return accu / count * 100.0


best_validation_accuracy = 0
best_epoch = -1
val_list = [0]*10

for epoch in range(1000):
    
    ## Training
    for i in range(0, len(train_indices), batch_size):
        images = train_validation_set.images[train_indices[i: i+batch_size]] 
        labels = train_validation_set.labels[train_indices[i: i+batch_size]]
        error = lib.train_one_step(model, loss, learning_rate, images, labels)
    
    validation_accuracy = validate()

   
    print("Epoch %d: loss: %f, validation accuracy: %.2f%%" % (epoch, error, validation_accuracy))

    val_list.append(validation_accuracy)
    if(len(val_list) > 10):
        val_list.pop(0)

    if  np.argmax(val_list) == 0: 
        best_epoch = epoch
        best_validation_accuracy = val_list.pop(0)
        print("Stopping...")
        break

print("Test set performance: %.2f%%" % test())