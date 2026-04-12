import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.architecture import ISLModel
import matplotlib.pyplot as plt

SIGNS = ['hello', 'yes', 'no', 'help', 'thanks', 'good']
EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 0.001

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
X_train = np.load(os.path.join(BASE, 'data/X_train.npy'))
X_test  = np.load(os.path.join(BASE, 'data/X_test.npy'))
y_train = np.load(os.path.join(BASE, 'data/y_train.npy'))
y_test  = np.load(os.path.join(BASE, 'data/y_test.npy'))

X_train = torch.FloatTensor(X_train)
X_test  = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test  = torch.LongTensor(y_test)

train_ds = TensorDataset(X_train, y_train)
test_ds  = TensorDataset(X_test, y_test)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_dl  = DataLoader(test_ds, batch_size=BATCH_SIZE)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

model = ISLModel(num_classes=len(SIGNS)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

train_losses, test_losses, accuracies = [], [], []
best_acc = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_dl:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for X_batch, y_batch in test_dl:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            test_loss += criterion(output, y_batch).item()
            correct += (output.argmax(1) == y_batch).sum().item()

    acc = correct / len(y_test) * 100
    scheduler.step(test_loss)
    train_losses.append(total_loss / len(train_dl))
    test_losses.append(test_loss / len(test_dl))
    accuracies.append(acc)

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), '../isl_model.pt')

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} — Loss: {total_loss/len(train_dl):.4f} — Acc: {acc:.1f}%")

print(f"\nBest accuracy: {best_acc:.1f}%")
print("Model saved as isl_model.pt")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train loss')
plt.plot(test_losses, label='Test loss')
plt.legend()
plt.title('Loss')
plt.subplot(1, 2, 2)
plt.plot(accuracies)
plt.title('Accuracy')
plt.savefig('training_plot.png')
print("Plot saved as training_plot.png")