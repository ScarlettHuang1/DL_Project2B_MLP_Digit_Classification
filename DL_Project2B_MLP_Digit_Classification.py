#!/usr/bin/env python
# coding: utf-8

# # MNIST MLP Digit Recognition Network
# 
# We will code a basic digit recognition network. The data are images which specify the digits 1 to 10 as (1, 28, 28) data - this data is black and white images. Each pixed of the image is an intensity between 0 and 255, and together the (1, 28, 28) pixel image can be visualized as a picture of a digit. The data is given to you as $\{(x^{(i)}, y^{(i)})\}_{i=1}^{N}$ where $y$ is the given label and x is the (1, 28, 28) data. This data will be gotten from `torchvision`, a repository of computer vision data and models.
# 
# Highlevel, the model and notebook goes as follows:
# *   You first download the data and specify the batch size of B = 16. Each image will need to be turned from a (1, 28, 28) volume into a vector of dimension 784 = 1 * 28 * 28. So each batch will be of size (16, 784).
# *   Then, you pass the model through two hidden layers, one of dimension (784, 32) and another of dimension (32, 16). After each linear map, you pass the data through a TanH nonlinearity.
# *   Finally, you pass the data through a (32, 10) linear layer and you return the log softmax of the data.
# *   What objective do you use? Be careful!
# *   How do you compute accuracy both manually and with torchmetrics?
# *   How do you compute AUROC?
# 
# All asserts should pass and accuracy should be higher than 85%. Otheer nonlinearity, like ReLU, might get higher.

# In[1]:


get_ipython().system('pip install torchmetrics')
# !pip install torchvision


# In[2]:


import torchvision
from torchvision import transforms
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torchmetrics


# In[11]:


SEED = 1
torch.manual_seed(SEED)


# In[12]:


image_path = './'

# Use ToTensor to transform the data and scale it by 255
# Look up transforms and Compose as well

transform = transforms.Compose([transforms.ToTensor()])

mnist_train_dataset = torchvision.datasets.MNIST(
    root=image_path,
    train=True,
    transform=transform,
    download=True
  )

mnist_test_dataset = torchvision.datasets.MNIST(
    root=image_path,
    train=False,
    transform=transform,
    download=False
)


# In[13]:


BATCH_SIZE = 64
LR = 0.001
EPOCHS = 20
# Define the DL for train and test
train_dl = DataLoader(mnist_train_dataset, batch_size = BATCH_SIZE, shuffle=True)
test_dl = DataLoader(mnist_test_dataset, batch_size = BATCH_SIZE, shuffle=True)


# In[14]:


class MLPClassifier(nn.Module):

  def __init__(self):
    super().__init__()
    # Define the layers
    self.linear1 = nn.Linear(784, 32)
    self.linear2 = nn.Linear(32, 16)
    self.linear3 = nn.Linear(16, 10)

  def forward(self, x):
    # Flatten x to be of last dimension 784
    x = x.view(x.size(0),-1)

    # Pass through linear layer 1
    x = self.linear1(x)

    # Apply tanh
    x = torch.tanh(x)

    # Pass through linear layer 2
    x = self.linear2(x)

    # Apply tanh
    x = torch.tanh(x)

    # Pass through linear layer 3
    x = self.linear3(x)

    # Return the LogSoftmax of the data
    # This will affect the loss we choose below
    return nn.LogSoftmax(dim = 1)(x)

model = MLPClassifier()


# In[15]:


from torchmetrics.classification import Accuracy,AUROC ##

# Get the loss function; remember you are outputting the LogSoftmax so be careful what loss you pick
loss_fn = nn.NLLLoss()

# Set the optimizer to SGD and let the learning rate be LR
optimizer = torch.optim.SGD(model.parameters(), lr=LR)

torch.manual_seed(SEED)
for epoch in range(EPOCHS):
    accuracy_hist_train = 0
    auroc_hist_train = 0.0
    loss_hist_train = 0
    # Loop through the x and y pairs of data
    for x_batch, y_batch in train_dl:
        # Get the model predictions
        pred = model(x_batch) 

        # Get the loss
        loss = loss_fn(pred,y_batch)

        # Get the gradients
        loss.backward()

        # Add to the loss
        # Note that loss is a mean over the batch size and we need the total sum over the number of samples in the dataset
        loss_hist_train += loss.item()

        # Update the prameters
        optimizer.step()

        # Zero out the gradient
        optimizer.zero_grad()

        # Get the number of correct predictions, 
        # This should be a tensor
        is_correct_1 = (torch.argmax(pred, dim = 1) == y_batch).float()

        # Get the number of correct predictions, do this with torchmetrics
        # This should be a Float
        is_correct_2 = Accuracy(task='multiclass',num_classes=10)(pred.argmax(dim=1), y_batch) * len(y_batch)

        assert(is_correct_1.sum() == is_correct_2)

        accuracy_hist_train += is_correct_1.sum() 

        # Get the AUROC - make sure to multiply by the batch length since this is just the AUC over the batch and you want to take a weighted average later
        auroc_hist_train += AUROC(task="multiclass", num_classes=10)(pred,y_batch) * BATCH_SIZE
        
    accuracy_hist_train /= len(train_dl.dataset)
    auroc_hist_train /= len(train_dl.dataset)
    loss_hist_train /= len(train_dl.dataset)
    print(f'Train Metrics Epoch {epoch} Loss {loss_hist_train:.4f} Accuracy {accuracy_hist_train:.4f} AUROC {auroc_hist_train:.4f}')

    accuracy_hist_test = 0
    auroc_hist_test = 0.0
    loss_hist_test = 0.0
    # Get the average value of each metric across the test batches
    # Add a "with" clause here so that no gradients are computed; we want to just evaluate the model
    with torch.no_grad():
      accuracy_hist_test = 0
      auroc_hist_test = 0.0
      # Loop through the x and y pairs of data
      for x_batch, y_batch in test_dl:
          # Get he the model predictions
          pred = model(x_batch) ##

          # Get the loss
          loss = loss_fn(pred, y_batch)

          # Add to the loss
          # Note that loss is a mean over the batch size and we need the total sum over the number of samples in the dataset
          loss_hist_test += loss.item()

          # Get the number of correct predictions via torchmetrics
          is_correct = Accuracy(task='multiclass',num_classes=10)(pred.argmax(dim=1), y_batch)* len(y_batch)

          # Get the accuracy
          accuracy_hist_test += is_correct

          # Get AUROC
          auroc_hist_test += AUROC(task="multiclass", num_classes=10)(pred,y_batch) * BATCH_SIZE
      # Normalize the metrics by the right number
      accuracy_hist_test /= len(test_dl.dataset)
      auroc_hist_test /= len(test_dl.dataset)
      loss_hist_test /= len(test_dl.dataset)
      print(f'Test Metrics Epoch {epoch} Loss {loss_hist_test:.4f} Accuracy {accuracy_hist_test:.4f} AUROC {auroc_hist_test:.4f}')


# In[16]:


# Get train/test final accuracy directly; normalize the data by 255.0
# Should be around 85%

train_x = torch.stack([torch.tensor(tup[0]) for tup in mnist_train_dataset])
train_y = torch.stack([torch.tensor(tup[1]) for tup in mnist_train_dataset])

pred = model((train_x))
is_correct = (torch.argmax(pred, dim = 1) == train_y).float()
print(f'Total Final Test accuracy: {is_correct.mean():.4f}')

test_x = torch.stack([torch.tensor(tup[0]) for tup in mnist_test_dataset])
test_y = torch.stack([torch.tensor(tup[1]) for tup in mnist_test_dataset])

pred = model((test_x))
is_correct = (torch.argmax(pred, dim = 1) == test_y).float()
print(f'Total Final Test accuracy: {is_correct.mean():.4f}')


# In[ ]:




