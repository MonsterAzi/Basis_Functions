import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from dataclasses import dataclass
# from fvcore.nn import FlopCountAnalysis

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class Config:
    batch_size: int = 64
    learning_rate: float = 0.002
    epochs: int = 20
    patience: int = 3
    num_workers: int = 1
    rotation: int = 3
    translation: float = 0.06
    shear_angle: int = 0.6

def vieta_pell(n, x):
    if n == 0:
        return 2 * torch.ones_like(x)
    elif n == 1:
        return x
    else:
        return x * vieta_pell(n - 1, x) + vieta_pell(n - 2, x)

class VietaPellKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(VietaPellKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree
        self.vp_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.vp_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))

    def forward(self, x):
        # Normalize x to [-1, 1] using tanh
        x = torch.tanh(x)

        # Compute the Vieta-Pell basis functions
        vp_basis = []
        for n in range(self.degree + 1):
            vp_basis.append(vieta_pell(n, x))
        vp_basis = torch.stack(vp_basis, dim=-1)  # shape = (batch_size, input_dim, degree + 1)

        # Compute the Vieta-Pell interpolation
        y = torch.einsum("bid,iod->bo", vp_basis, self.vp_coeffs)  # shape = (batch_size, output_dim)
        y = y.view(-1, self.output_dim)

        return y

class MNISTVietaPellKAN(nn.Module):
    def __init__(self):
        super(MNISTVietaPellKAN, self).__init__()
        self.trigkan1 = VietaPellKANLayer(784, 32, 3)
        self.bn1 = nn.LayerNorm(32)
        self.trigkan2 = VietaPellKANLayer(32, 32, 3)
        self.bn2 = nn.LayerNorm(32)
        self.trigkan3 = VietaPellKANLayer(32, 10, 3)

    def forward(self, x):
        x = x.view(-1, 28*28)
        #x=x.tanh()
        x = self.trigkan1(x)
        x = self.bn1(x)
        x = self.trigkan2(x)
        x = self.bn2(x)
        x = self.trigkan3(x)
        return x

transform_train = v2.Compose([
    v2.ToImage(),
    v2.RandomAffine(degrees=Config.rotation, translate=(Config.translation, Config.translation), shear=Config.shear_angle),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.1307,), (0.3081,))
])

transform_test = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=Config.num_workers)
test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=Config.num_workers)

num_classes = 10


criterion = nn.CrossEntropyLoss()



def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    for data, target in progress_bar:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        progress_bar.set_postfix({'Loss': total_loss / (progress_bar.n + 1), 'Acc': 100. * correct / total})
    
    return total_loss / len(train_loader), correct / total

def validate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    return total_loss / len(test_loader), correct / total

Model_Name='VietaPell'  #Add names of other models
model0 = MNISTVietaPellKAN().to(device)
model=model0
total_params = sum(p.numel() for p in model0.parameters() if p.requires_grad)
# flops = FlopCountAnalysis(model, inputs=(torch.randn(1, 28 * 28).to(device),)).total()
print(f"Total trainable parameters of {Model_Name}: {total_params}")
# print(f"FLOPs of {Model_Name}: {flops}")

def train_and_validate(model, train_loader, test_loader, criterion, optimizer, device, epochs, patience):
    best_test_loss = float('inf')
    best_weights = None
    no_improve = 0
    
    for epoch in range(epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = validate(model, test_loader, criterion, device)
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_weights = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve == patience:
                print(f'Early stopping after {epoch+1} epochs')
                break
    
    return best_weights, best_test_loss


optimizers = optim.Adam(model.parameters(), lr=Config.learning_rate)

best_weights, model_times = train_and_validate(model, train_loader, test_loader, criterion, optimizers, device, Config.epochs, Config.patience)

# Save the best weights for model
model.load_state_dict(best_weights)
torch.save(model.state_dict(), f'{Model_Name}_best_weights.pth')

# Print the processing time for model
print(f"{Model_Name} processing time: {model_times:.2f} seconds")

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='weighted')
    kappa = cohen_kappa_score(all_targets, all_preds)
    
    # Print classification report
    print(classification_report(all_targets, all_preds))
    
    # Plot confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    return accuracy, f1, kappa

# Usage
model.load_state_dict(best_weights)
accuracy, f1, kappa = evaluate_model(model, test_loader, device)
print(f'Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, Cohen\'s Kappa: {kappa:.4f}')