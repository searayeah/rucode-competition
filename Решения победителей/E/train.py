import os
import shutil
import pandas as pd
from datasets import load_dataset
from transformers import CLIPModel, CLIPProcessor, TrainingArguments, Trainer, AutoTokenizer
from torch import nn
from sklearn.model_selection import train_test_split
import time
import copy
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
from torchvision import transforms


class CustomCLIPModel(nn.Module):
    def __init__(self, clip_bb):
        super().__init__()
        self.bb = clip_bb
        self.fc1 = nn.Linear(1024, 512)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(512, 16)
 
    def forward(self, images=None):
        outputs = self.bb(images).last_hidden_state[:, -1]
        outputs = self.act(self.fc1(outputs))
        outputs = self.fc2(outputs)
        return outputs
    
model_name = "zer0int/CLIP-GmP-ViT-L-14"
clip_bb = CLIPModel.from_pretrained(model_name).vision_model
tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
model = CustomCLIPModel(clip_bb)
processor = CLIPProcessor.from_pretrained(model_name)


class ImageFolderDataset(Dataset):
    def __init__(self, root_dir1, root_dir2, transform=processor.image_processor, use_pl=False):
        self.root_dir1 = root_dir1
        self.root_dir2 = root_dir2
        self.transform = transform
        self.classes = sorted(list(map(int, os.listdir(root_dir1))))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.images = []
        self.labels = []
        
        for root_dir in [root_dir1, root_dir2]:
            for class_name in self.classes:
                class_path = os.path.join(root_dir, f'{class_name}')
                for img_path in os.listdir(class_path):
                    img = Image.open(os.path.join(class_path, img_path)).convert('RGB')
                    tensor = self.to_tensor(img)
                    self.images.append(tensor)
                    self.labels.append(class_name-1)
        if use_pl:
            self.df3 = pd.read_csv('pl.csv')
            for el in range(len(self.df3)):
                img_name, label = self.df3.iloc[el]
                img = Image.open(os.path.join('pretrained_dataset', img_name)).convert('RGB')
                tensor = self.to_tensor(img)
                self.images.append(tensor)
                self.labels.append(label-1)

    def to_tensor(self, img):
        if isinstance(self.transform, list):
            for t in self.transform:
                img = t(img)
        elif callable(self.transform):
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = torch.tensor(self.labels[index])
        return image['pixel_values'][0], label


def train_model(datasets, dataloaders, model, criterion, optimizer, scheduler, num_epochs, device):
    since = time.time()
    
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-' * 10)
        
        for phase in ['train']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0.0
            running_total = 0.0
            
            for step, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
               
                # Zero out the grads
                optimizer.zero_grad()
                
                # Forward
                # Track history in train mode
                with torch.set_grad_enabled(phase == 'train'):
                    model = model.to(device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1) 
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                running_total += len(labels.data)
                
                if (step + 1) % 100 == 0:
                    print(f'[{step + 1}/{len(dataloaders[phase])}].')
                    print(f'Loss {running_loss / running_total}. Accuracy {running_corrects / running_total}')
            
                
            epoch_loss = running_loss / len(datasets[phase])
            epoch_acc = running_corrects.double() / len(datasets[phase])
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
        
        print()
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:.4f}'.format(best_acc))
    
    model.load_state_dict(model.state_dict())
    
    return model



# Настройка параметров
num_classes = 16  # Предполагается, что у вас 16 классов
batch_size = 8
num_epochs = 6
learning_rate = 1e-5

# Загрузка датасета
train_dataset = ImageFolderDataset(root_dir1="dataset/train", root_dir2="dataset/test")


# Создание DataLoader'ов
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Определение критерия
criterion = nn.CrossEntropyLoss()

# Определение оптимизатора
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Определение планировщика обучения
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataloader = {"train" : train_dataloader}
datasets = {"train":train_dataset}

model = train_model(model = model, criterion = criterion, optimizer = optimizer, num_epochs= num_epochs, dataloaders=dataloader, datasets=datasets, device=device, scheduler=scheduler)
torch.save(model.state_dict(), 'my_model.pt')