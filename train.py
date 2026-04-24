import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import cv2
from model import DrowsinessCNN
import glob

# --- Configuration ---
DATA_DIR = './dataset' # Users need to put data here: ./dataset/Closed and ./dataset/Open
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
MODEL_SAVE_PATH = 'drowsines_model.pth'

class EyeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = [] # 0: Closed, 1: Open
        
        # Load Closed images
        closed_path = os.path.join(root_dir, 'Closed')
        if os.path.exists(closed_path):
            files = glob.glob(os.path.join(closed_path, '*'))
            self.image_paths.extend(files)
            self.labels.extend([0] * len(files))
            
        # Load Open images
        open_path = os.path.join(root_dir, 'Open')
        if os.path.exists(open_path):
            files = glob.glob(os.path.join(open_path, '*'))
            self.image_paths.extend(files)
            self.labels.extend([1] * len(files))
            
        print(f"Found {len(self.image_paths)} images.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use PIL for transforms if needed, or convert manually
        # Here we just assume simple resizing logic if not using complex transforms
        # But ToPILImage needs RGB or Grayscale with C,H,W? No, numpy array H,W,C
        
        # Simple read for consistent processing
        if self.transform:
            image = self.transform(image)
            
        label = self.labels[idx]
        return image, label

def train():
    if not os.path.exists(DATA_DIR) or (not os.path.exists(os.path.join(DATA_DIR, 'Open'))):
        print(f"ERROR: Dataset not found in {DATA_DIR}")
        print("Please create folders './dataset/Open' and './dataset/Closed' and put eye images there.")
        return

    # Transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((24, 24)),
        transforms.ToTensor(),
    ])

    dataset = EyeDataset(DATA_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = DrowsinessCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("Starting training...")
    
    for epoch in range(10): 
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/10, Loss: {running_loss/len(dataloader):.4f}, Accuracy: {accuracy:.2f}%")
        
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()
