# Import necessary libraries
import torch
from torch import nn
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import time
# Import the seaborn library for plotting
import seaborn as sns

df = pd.read_csv(r'Life Expectancy Data.csv')
df.head()
df.info()
# Checking the Missing Values
sns.heatmap(df.isnull(), yticklabels=False, cmap= "viridis")

#A summary of missing variables represented as a percentage of the total missing content. 
def missingness_summary(df, print_log=False, sort='ascending'):
  s = df.isnull().sum()*100/df.isnull().count()
  s = s [s > 0]
  if sort.lower() == 'ascending':
    s = s.sort_values(ascending=True)
  elif sort.lower() == 'descending':
    s = s.sort_values(ascending=False)  
  if print_log: 
    print(s)
  
  return pd.Series(s)

suspects = missingness_summary(df, True, 'descending')

selected_columns = ["Life expectancy ", "Adult Mortality", "Alcohol", "GDP"]

# new DataFrame object with the selected columns only
df_subset = df[selected_columns]

df_subset

df["Status"].value_counts()

# create mask for developed/developing country comparison
dev_mask = (df['Status'] == "Developed")

# Rearrange selected columns to a 2x2 matrix (as numpy array)

colname_matrix = df_subset.columns.to_numpy().reshape(2,2)
fig, axs = plt.subplots(2,2, figsize=(14, 7))

for row in range(2):
    for col in range(2):
        colname_tmp = colname_matrix[row, col]

        # urrent axis to subplot index
        plt.sca(axs[row, col])
        
        # bin edges and use histogram plotting function
        # of pandas dataframe
        bin_edges = np.linspace(df_subset[colname_tmp].min(),
                                df_subset[colname_tmp].max(), 50)
        df_subset[~dev_mask][colname_tmp].plot.hist(bins=bin_edges,
                                                    label="Developing",
                                                    density=True)

        df_subset[dev_mask][colname_tmp].plot.hist(histtype="step",
                                                   bins=bin_edges,
                                                   label="Developed",
                                                   density=True)

        plt.xlabel(colname_tmp)
        plt.legend(loc="upper right")

plt.show()
plt.close()

#  plots in 2x2 matrix
colname_matrix = df_subset.columns.to_numpy().reshape(2,2)
fig, axs = plt.subplots(2,2, figsize=(10, 7))

for row in range(2):
    for col in range(2):
        colname_tmp = colname_matrix[row, col]
        
        # set current axis to subplot index
        plt.sca(axs[row, col])
        
        df_subset[colname_tmp].plot.box()
        
plt.show()
plt.close()

df[["Country", "GDP"]].sort_values(by="GDP", ascending=False)

df.dtypes

df["Country"].value_counts()

# drop Country column for now
df = df.drop(columns=["Country"])
df["Status"].value_counts()
#Status has two unique values developed and developing

encoded_status = pd.get_dummies(df["Status"], prefix="status")

encoded_status
# drop columns
encoded_status = encoded_status.drop(columns=["status_Developing"])

# drop the original status columns from the WHO DataFrame
df = df.drop(columns=["Status"])

# add the new status_Developed column to the WHO DataFrame
df["status_Developed"] = encoded_status["status_Developed"]
#Object no longer contains dtypes
df.dtypes
# Nan or INF per column to count them
(~np.isfinite(df)).sum()
#checks for inf values
np.isinf(df).sum()
#we see that that there are no inf values which means all the tests are NaN
from sklearn.impute import KNNImputer

# use imputer
imputer = KNNImputer()

# transform the dataframe and save it into a new DataFrame object
df_clean = pd.DataFrame(imputer.fit_transform(df),
                            columns=df.columns)

(~np.isfinite(df_clean)).sum()
#We imputed all the NaN values using the kNN imputation approach

# Split the cleaned DataFrame into targets and features
targets = df_clean['Life expectancy ']
data = df_clean.drop(columns=['Life expectancy '])

# Split into training/validation/test sets according to 70/15/15 split
from sklearn.model_selection import train_test_split

# split off training set (70% of samples)
x_train, x_remain, y_train, y_remain = train_test_split(data, targets,
                                                        train_size=0.7,
                                                        random_state=42)


# split remainder into test and validation sets
# (15% each, corresponding to half of the remaining non-training samples)
x_val, x_test, y_val, y_test = train_test_split(x_remain, y_remain,
                                                train_size=0.5,
                                                random_state=42)

# numbers check out
n_full = data.shape[0]
train_percent = (x_train.shape[0]/n_full)*100
val_percent = (x_val.shape[0]/n_full)*100
test_percent = (x_test.shape[0]/n_full)*100

print(f"Train set corresponds to {train_percent:.2f}% of the full data.")
print(f"Validation set corresponds to {val_percent:.2f}% of the full data.")
print(f"Test set corresponds to {test_percent:.2f}% of the full data.")

# import standard scaler from scikit-learn
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# fit (i.e. do the mean/std computation based on the training set features)
# and transform training data
x_train = scaler.fit_transform(x_train).astype("float32")

x_val = scaler.transform(x_val).astype("float32")
x_test = scaler.transform(x_test).astype("float32")
y_train = y_train.astype("float32")
y_val = y_val.astype("float32")
y_test = y_test.astype("float32")
# checking
print("Means of training set:\n", x_train.mean(axis=0), "\n")
print("Standard deviations of training set:\n", x_train.std(axis=0), "\n\n")

print("Means of validation set:\n", x_val.mean(axis=0), "\n")
print("Standard deviations of validation set:\n", x_val.std(axis=0), "\n\n")

print("Means of test set:\n", x_test.mean(axis=0), "\n")
print("Standard deviations of test set:\n", x_test.std(axis=0), "\n\n")

class net (nn.Module):
    def __init__(self, layers, n_inputs=5):
        super().__init__()
        
        #linear layers activation
        self.layers=[]
        for nodes in layers:
            self.layers.append(nn.Linear(n_inputs,nodes))
            self.layers.append(nn.ReLU())
            n_inputs= nodes
            
        self.layers.append(nn.Linear(n_inputs, 1))
        # build pytorch model as sequence of our layers
        self.model_stack=nn.Sequential(*self.layers)
        
    def forward (self, x):
        # the forward call just takes data (x) and sends it through the model
        # to produce an output
        return self.model_stack(x)
    
    def predict(self, x):
        
        with torch.no_grad():
            self.eval()
            x = torch.tensor(x)
            prediction = self.forward(x).detach().cpu().numpy()
        return prediction
        
    
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
import torch.nn.functional as F

train_set = TensorDataset(torch.tensor(x_train),
                          torch.from_numpy(y_train.to_numpy()).reshape(-1, 1))
val_set = TensorDataset(torch.tensor(x_val),
                        torch.from_numpy(y_val.to_numpy()).reshape(-1, 1))

# DataLoader objects
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32)

# set number of epochs
epochs = 10

# build model using a single layer with 64 neurons
reg_model = net(layers=[64], n_inputs = x_train.shape[1])

optimizer = optim.Adam(reg_model.parameters(), lr=1e-3)

# Define loss function.
loss = F.mse_loss

#empty lists for storage of training and validation losses
train_losses = []
val_losses = []

start = time.time()

# outer training loop
for epoch in range(epochs):
    
    running_train_loss = 0
    
    # 
    reg_model.train()

    # training part of outer loop = inner loop
    for batch in train_loader:
        
        data, targets = batch
        output = reg_model(data)
        tmp_loss = loss(output, targets)
        optimizer.zero_grad()
        tmp_loss.backward()
        optimizer.step()
        
        running_train_loss += tmp_loss.item()
    
    print(f"Train loss after epoch {epoch+1}: {running_train_loss/len(train_loader)}")
    train_losses.append(running_train_loss/len(train_loader))
    
    ## validation part of outer loop
    
    running_val_loss = 0
    # deactivate gradient computation
    with torch.no_grad():
        
        # set model to evaluation mode
        reg_model.eval()
        
        # loop over validation DataLoader
        for batch in val_loader:
            
            data, targets = batch
            output = reg_model(data)
            tmp_loss = loss(output, targets)
            running_val_loss += tmp_loss.item()
        
        mean_val_loss = running_val_loss/len(val_loader)
        print(f"Validation loss after epoch {epoch+1}: {mean_val_loss}")
        
        # If the validation loss of the model is lower than that of all the
        # previous epochs, save the model state
        if epoch == 0:
            torch.save(reg_model.state_dict(), "./min_val_loss_reg_model.pt")
        elif (epoch > 0) and (mean_val_loss < np.min(val_losses)):
            print("Lower loss!")
            torch.save(reg_model.state_dict(), "./min_val_loss_reg_model.pt")
        
        val_losses.append(mean_val_loss)

end = time.time()
print(f"Done training {epochs} epochs!")
print(f"Training took {end-start:.2f} seconds!")

plt.plot(np.arange(epochs), train_losses, label="training")
plt.plot(np.arange(epochs), val_losses, label="validation")
plt.ylabel("loss")
plt.xlabel("epochs")
plt.legend(loc="upper right")
plt.show()
plt.close()

#Here we trained for only 10 epochs, we can train a little longer to improve results

from sklearn.metrics import (mean_absolute_error,
                             mean_squared_error,
                             median_absolute_error,
                             max_error, r2_score)

# minimum validation loss model
reg_model.load_state_dict(torch.load("./min_val_loss_reg_model.pt"))

y_pred = reg_model.predict(torch.tensor(x_test))

print("Classification performance report")

print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"Median Absolute Error: {median_absolute_error(y_test, y_pred):.2f}")
print(f"Max Error: {max_error(y_test, y_pred):.2f}")
print(f"R2 score: {r2_score(y_test, y_pred):.2f}")

plt.errorbar(y_test, y_pred, fmt='bo', label="True values")
plt.xlabel("True Life Expectancy")
plt.ylabel("Predicted Life Expectancy")
plt.legend(loc="upper right")
plt.show()
plt.close()