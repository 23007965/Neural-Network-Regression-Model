# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: PARTHIBAN
### Register Number: 212223230145
```python
import pandas as pd
data=pd.read_csv("/content/height_weight.csv")
data.columns = data.columns.str.strip()
data

x=data[['height']]
y=data[['weight']]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=33)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

import torch

x_train_tensor=torch.tensor(x_train,dtype=torch.float32)
y_train_tensor=torch.tensor(y_train.values,dtype=torch.float32).view(-1,1)
x_test_tensor=torch.tensor(x_test,dtype=torch.float32)
y_test_tensor=torch.tensor(y_test.values,dtype=torch.float32).view(-1,1)

import torch.nn as nn

class NeuralNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1=nn.Linear(1,8)
    self.fc2=nn.Linear(8,10)
    self.fc3=nn.Linear(10,1)
    self.relu=nn.ReLU()
    self.history={'loss':[]}

  def forward(self,x):
    x=self.relu(self.fc1(x)) # Apply relu after the first linear layer
    x=self.relu(self.fc2(x)) # Apply relu after the second linear layer
    x=self.fc3(x)
    return x

Parthiban = NeuralNet()
criterion=nn.MSELoss()
optimizer=torch.optim.RMSprop(Parthiban.parameters(),lr=0.001)

def train_model(Parthiban,x_train,y_train,criterion,optimizer,epochs=2000):
  for epoch in range(epochs):
    optimizer.zero_grad()
    loss=criterion(Parthiban(x_train),y_train)
    loss.backward()
    optimizer.step()

    Parthiban.history['loss'].append(loss.item())
    if epoch % 200 ==0:
      print(f"Epoch [{epoch}/{epochs}], loss: {loss.item():.6f}")

train_model(Parthiban,x_train_tensor,y_train_tensor,criterion,optimizer)

with torch.no_grad():
  test_loss=criterion(Parthiban(x_test_tensor),y_test_tensor)
  print(f"Test Loss: {test_loss.item():.6f}")

import matplotlib.pyplot as plt
plt.plot(Parthiban.history['loss'])
plt.title('Loss during Training')
plt.xlabel('Epochs')
plt.ylabel('Loss')

X_n1_1 = torch.tensor([[105]], dtype=torch.float32)
prediction = Parthiban(torch.tensor(scaler.transform(X_n1_1), dtype=torch.float32)).item()
print(f'Prediction: {prediction}')

```
## Dataset Information

<img width="549" height="685" alt="image" src="https://github.com/user-attachments/assets/8c0ca5b3-772b-4f23-89f3-b6a362e578c4" />


## OUTPUT

### Training Loss Vs Iteration Plot

<img width="779" height="600" alt="image" src="https://github.com/user-attachments/assets/7a28c022-d7c4-486a-9294-e707a1615913" />


### New Sample Data Prediction

Include your sample input and output here

## RESULT

The program to develop a neural network regression model for the given dataset has been executed successively
