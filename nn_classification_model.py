from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from data_processing import get_mel_loaders

class PrimaryCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(PrimaryCNN, self).__init__()
        self.name = "CNN"
        
        # Conv Block 1: Input (B, 1, 128, T) -> Output (B, 32, 64, T/2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))
        
        # Conv Block 2: Output (B, 64, 32, T/4)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))
        
        # Conv Block 3: Output (B, 128, 16, T/8)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        
        # Global Average Pooling & Fully Connected Classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes))

    def forward(self, img):
        x = self.conv1(img)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.global_pool(x)
        logits = self.classifier(x)
        return logits

def train_model(model, train_loader, val_loader, batch_size=32, learning_rate=0.005, weight_decay=0.001, num_epochs=30):

    torch.manual_seed(30)

    criterion = nn.CrossEntropyLoss() # Multiclass classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    metrics = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": []}

    best_val_acc = 0.0
    best_model_weights = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for i, data in enumerate(train_loader, 0):
            mel_spec, labels = data
            
            optimizer.zero_grad()
            outputs = model(mel_spec)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * mel_spec.size(0)
            preds = outputs.argmax(dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
            
        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train

        val_acc, val_loss = evaluate(model, val_loader, criterion)

        metrics["train_loss"].append(epoch_train_loss)
        metrics["train_acc"].append(epoch_train_acc)
        metrics["val_loss"].append(val_loss)
        metrics["val_acc"].append(val_acc)

        print(("Epoch {}: Train acc: {:.4f}, Train loss: {:.4f} | " +
           "Validation acc: {:.4f}, Validation loss: {:.4f}").format(
               epoch + 1,
               metrics["train_acc"][epoch],
               metrics["train_loss"][epoch],
               metrics["val_acc"][epoch],
               metrics["val_loss"][epoch]))

        model_path = get_model_name(model.name, batch_size, learning_rate, epoch) 
        torch.save(model.state_dict(), model_path)
    
    print("Finished Training")
    epochs = np.arange(1, num_epochs + 1)
    np.savetxt("{}_train_acc.csv".format(model_path), metrics["train_acc"])
    np.savetxt("{}_train_loss.csv".format(model_path), metrics["train_loss"])
    np.savetxt("{}_val_acc.csv".format(model_path), metrics["val_acc"])
    np.savetxt("{}_val_loss.csv".format(model_path), metrics["val_loss"])

    plot_training_curve(model_path)

def evaluate(model, val_loader, criterion):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_epoch = 0

    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            mel_spec, labels = data

            outputs = model(mel_spec)
            loss = criterion(outputs, labels)

            # Find the predicted class (the index with the highest logit score)
            _, predicted = torch.max(outputs.data, 1)

            # Count accuracy where prediction matches the true label
            total_acc += (predicted == labels).sum().item()
            total_loss += loss.item()
            total_epoch += labels.size(0)

    # Calculate accuracy and loss
    acc = float(total_acc)/total_epoch
    loss = float(total_loss) / (i+1)

    return acc, loss

# ---- THE FOLLOWING CODE IS USED FROM APS360 LAB 2 ----
def get_model_name(name, batch_size, learning_rate, epoch):
    """ Generate a name for the model consisting of all the hyperparameter values
    Args:
        config: Configuration object containing the hyperparameters
    Returns:
        path: A string with the hyperparameter name and value concatenated
    """
    path = "model_{0}_bs{1}_lr{2}_epoch{3}".format(name, batch_size, learning_rate, epoch)

    return path

def plot_training_curve(path):
    """ Plots the training curve for a model run, given the csv files
    containing the train/validation accuracy/loss.

    Args:
        path: The base path of the csv files produced during training
    """
    train_acc = np.loadtxt("{}_train_acc.csv".format(path))
    val_acc = np.loadtxt("{}_val_acc.csv".format(path))

    train_loss = np.loadtxt("{}_train_loss.csv".format(path))
    val_loss = np.loadtxt("{}_val_loss.csv".format(path))

    plt.title("Train vs Validation Accuracy")
    n = len(train_acc) # number of epochs
    plt.plot(range(1,n+1), train_acc, label="Train")
    plt.plot(range(1,n+1), val_acc, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.show()

    plt.title("Train vs Validation Loss")
    plt.plot(range(1,n+1), train_loss, label="Train")
    plt.plot(range(1,n+1), val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()


batch_size = 32  # Hyperparameter to adjust

train_loader, val_loader, test_loader, genres = get_mel_loaders(npz_path="processed_data/mel_splits.npz", batch_size=batch_size)

model = PrimaryCNN(num_classes=len(genres))
lr = 0.001
weight_decay = 0.001
num_epochs = 30

train_model(model, train_loader, val_loader)