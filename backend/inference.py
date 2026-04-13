import numpy as np
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import sys
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE)

from model.architecture import ISLModel

SIGNS = ['hello', 'yes', 'no', 'help', 'thanks', 'good']

model = ISLModel(num_classes=len(SIGNS))
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model.load_state_dict(torch.load(os.path.join(BASE, 'isl_model.pt'), map_location='cpu'))
model.eval()

def predict(sequence: np.ndarray):
    x = torch.FloatTensor(sequence).unsqueeze(0)
    with torch.no_grad():
        output = model(x)
        probs = torch.softmax(output, dim=1)
        confidence, predicted = probs.max(1)
    return SIGNS[predicted.item()], round(confidence.item() * 100, 1)