#
import torch
import torch.nn as nn
import torch.nn.functional as F

# MLP class
class MLP(nn.Module):
    # network
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(42, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, 100)
        self.fc5 = nn.Linear(100, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x= self.fc5(x)

        return x


mlp=MLP()
model_path = '../data/model_data/model1.pth'
mlp.load_state_dict(torch.load(model_path))


def discrimination(input_x):
  outputs = mlp(input_x)
  print(outputs)
  tensor_label = outputs.argmax(dim=1, keepdim=True)
  out_label = tensor_label.item()
  return out_label


in_arr=[[146,205,124,186,119,165,128,153,138,154,144,145,138,139,134,147,135,154,158,151,148,145,143,152,143,160,166,161,157,157,150,163,149,169,172,171,162,168,156,173,156,178]]
in_torch = torch.tensor(in_arr,dtype=torch.float32)
out=discrimination(in_torch)
print("out_label:{}".format(out))