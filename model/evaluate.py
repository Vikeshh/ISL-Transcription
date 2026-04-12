import numpy as np
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.architecture import ISLModel
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

SIGNS = ['hello', 'yes', 'no', 'help', 'thanks', 'good']
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

X_test = torch.FloatTensor(np.load(os.path.join(BASE, 'data/X_test.npy')))
y_test = np.load(os.path.join(BASE, 'data/y_test.npy'))

model = ISLModel(num_classes=len(SIGNS))
model.load_state_dict(torch.load('isl_model.pt', map_location='cpu'))
model.eval()

with torch.no_grad():
    outputs = model(X_test)
    preds = outputs.argmax(1).numpy()

print(classification_report(y_test, preds, target_names=SIGNS))

cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=SIGNS, yticklabels=SIGNS)
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
print("Saved confusion_matrix.png")