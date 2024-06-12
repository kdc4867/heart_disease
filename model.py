import torch
import torch.nn as nn

class HeartDiseaseModel(nn.Module):
    def __init__(self, input_dim):
        super(HeartDiseaseModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)

def load_model(file_path, input_dim):
    model = HeartDiseaseModel(input_dim)
    model.load_state_dict(torch.load(file_path))
    return model
