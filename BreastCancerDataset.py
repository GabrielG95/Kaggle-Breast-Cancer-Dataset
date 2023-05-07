import matplotlib.pyplot as plt
import torch
from torchvision.transforms import ToTensor
from torch import nn
import torchvision
import pandas as pd
from torchvision import datasets
from sklearn.datasets import load_breast_cancer
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix
import requests
import seaborn as sns
from pathlib import Path
from helper_functions import accuracy_fn

class NN(nn.Module):

    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int):
        super().__init__()
        self.linear_layer_1 = nn.Sequential(
                nn.Linear(in_features=input_shape,
                          out_features=hidden_units),
                nn.LeakyReLU(),
                nn.Linear(in_features=hidden_units,
                          out_features=hidden_units),
                nn.LeakyReLU()
                )
        self.linear_layer_2 = nn.Sequential(
                nn.Linear(in_features=hidden_units,
                          out_features=hidden_units),
                nn.LeakyReLU(),
                nn.Linear(in_features=hidden_units,
                          out_features=output_shape),
                nn.LeakyReLU()
                )

    def forward(self, x):
        return self.linear_layer_2(torch.sigmoid(self.linear_layer_1(x)))

# Download helper functions from Learn PyTorch repo
if Path('helper_functions.py').is_file():
  print('helper_functions.py already exists, skipping download')
else:
  print('Downloading helper_functions.py')
  request = requests.get('https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py')
  with open('helper_functions.py', 'wb') as f:
    f.write(request.content)

# Load data in
data = load_breast_cancer()
X = data['data']
y = data['target']

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# Create a TensorDataset
# This allows us to work with our data in batches.
# Otherwise we can't because we have four different dataset splits, we need 2 for batches.
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=32,
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_dataset,
                             batch_size=32,
                             shuffle=False)

# Visualize litte samples of data
df = pd.DataFrame({"Feature: 0": X_train[:,0],
                   'Feature: 1': X_train[:,1],
                   'Label': y_train})

# Standardize data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

class_names = data.target_names
features = data.feature_names
print(f'Number of features we have: {len(features)}')
print(f'Number of labels we have: {len(class_names)}')

print(f'\nX_train shape: {X_train.shape}\ny_trian shape: {y_train.shape}')
print(f'X_test shape: {X_test.shape}\ny_test shape: {y_test.shape}')

torch.manual_seed(42)
model_1 = NN(input_shape=X_train.shape[1],
             hidden_units=64,
             output_shape=2)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_1.parameters(),
                            lr=0.001)

# Make a list of our accuracy to plot
train_acc_list = []
test_acc_list = []

# Train function
def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.dataloader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn):
  train_loss, train_acc = 0, 0
  for batch, (X,y) in enumerate(dataloader):
      model.train()
      y_pred = model(X)
      loss = loss_fn(y_pred, y)
      acc = accuracy_fn(y, torch.sigmoid(y_pred).argmax(dim=1))
      train_acc += acc
      train_loss += loss
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

  # get average
  train_loss /= len(dataloader)
  train_acc /= len(dataloader)
  train_acc_list.append(train_acc)
  print(f'Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%')

# test function  
def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.dataloader,
              loss_fn,
              accuracy_fn):
  test_loss, test_acc = 0, 0
  model.eval()
  with torch.inference_mode():
    for X, y in dataloader:
      test_pred = model(X)
      loss = loss_fn(test_pred, y)
      acc = accuracy_fn(y, torch.sigmoid(test_pred).argmax(dim=1))
      test_acc += acc
      test_loss += loss

    # get average
    test_loss /= len(dataloader)
    test_acc /= len(dataloader)
    test_acc_list.append(test_acc)
  print(f'Test loss: {test_loss:.4f} | Test acc: {test_acc:.2f}%')

epochs = 31
for epoch in range(epochs):
    print(f'Epochs: {epoch}\n--------------')
    train_step(model=model_1,
               dataloader=train_dataloader,
               optimizer=optimizer,
               loss_fn=loss_fn,
               accuracy_fn=accuracy_fn)

    test_step(model=model_1,
              dataloader=test_dataloader,
              loss_fn=loss_fn,
              accuracy_fn=accuracy_fn)

# visualize predictions and true values
y_pred = []
for X, y in test_dataloader:
    with torch.inference_mode():
        model_1.eval()
        y_pred.extend(model_1(X).argmax(dim=1).tolist())

# create confusion matrix 
cm = confusion_matrix(y_test, y_pred)

# plot cm as heatmap
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()











