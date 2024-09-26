import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import time
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score, classification_report, confusion_matrix
from dataclasses import dataclass
from fvcore.nn import FlopCountAnalysis
import wandb
import seaborn as sns
from soap import SOAP
import math

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class Config:
    batch_size: int = 128
    learning_rate: float = 7e-4
    epochs: int = 50
    patience: int = 10
    num_workers: int = 10
    rotation: int = 5
    translation: float = 0.1
    shear_angle: int = 1
    betas: tuple[float,float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 1e-2
    precondition_frequency: int = 4

# Initialize wandb
run = wandb.init(
    # Set the project where this run will be logged
    project="mnist",
    # Track hyperparameters and run metadata
    config={
        "batch_size": Config.batch_size,
        "learning_rate": Config.learning_rate,
        "epochs": Config.epochs,
        "patience": Config.patience,
        "num_workers": Config.num_workers,
        "rotation": Config.rotation,
        "translation": Config.translation,
        "shear_angle": Config.shear_angle
    },
)

class HigherOrderLayer(nn.Module):
    def __init__(self, input_size, output_size, degree):
        super(HigherOrderLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.degree = degree

        # Create a single weight matrix for all degrees
        self.weight = nn.Parameter(torch.Tensor(output_size, input_size * (degree + 1)))
        self.bias = nn.Parameter(torch.Tensor(output_size))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        batch_size = x.size(0)
        
        # Compute powers of x up to degree
        x_powers = [x]
        for d in range(2, self.degree + 1):
            x_powers.append(torch.pow(x, d))
        
        # Concatenate all powers
        x_concat = torch.cat([torch.ones(batch_size, 1, device=x.device)] + x_powers, dim=1)
        
        # Perform the higher-order transformation
        output = F.linear(x_concat, self.weight, self.bias)
        
        return output

class MNISTVietaPellKAN(nn.Module):
    def __init__(self):
        super(MNISTVietaPellKAN, self).__init__()
        self.trigkan1 = HigherOrderLayer(784, 32, 3)
        self.bn1 = nn.LayerNorm(32)
        self.trigkan2 = HigherOrderLayer(32, 24, 4)
        self.bn2 = nn.LayerNorm(24)
        self.trigkan3 = HigherOrderLayer(24, 10, 3)

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



def train(model, train_loader, criterion, optimizer, scheduler, device):
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
        scheduler.step()
        
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
flops = FlopCountAnalysis(model, inputs=(torch.randn(1, 28 * 28).to(device),)).total()
print(f"Total trainable parameters of {Model_Name}: {total_params}")
print(f"FLOPs of {Model_Name}: {flops}")

def train_and_validate(model, train_loader, test_loader, criterion, optimizer, scheduler, device, epochs, patience):
    best_test_loss = float('inf')
    best_weights = None
    no_improve = 0
    total_time = 0
    
    for epoch in range(epochs):
        start_time = time.time()
        
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, scheduler, device)
        test_loss, test_acc = validate(model, test_loader, criterion, device)
        
        epoch_time = time.time() - start_time
        total_time += epoch_time

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "epoch_time": epoch_time
        }, step=epoch+1)
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        
        last_epoch = epoch
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_weights = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve == patience:
                print(f'Early stopping after {epoch+1} epochs')
                break
    
    return best_weights, total_time / (last_epoch + 1)


optimizer = SOAP(model.parameters(), lr=Config.learning_rate, betas=Config.betas, eps=Config.eps,
                 weight_decay=Config.weight_decay, precondition_frequency=Config.precondition_frequency)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=Config.learning_rate, epochs=Config.epochs, steps_per_epoch=469)

best_weights, model_times = train_and_validate(model, train_loader, test_loader, criterion, optimizer, scheduler, device, Config.epochs, Config.patience)

wandb.finish()

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
