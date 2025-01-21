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


# Функция для загрузки и обработки изображения
def load_and_process_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")["pixel_values"].to(device)
    return inputs

# Функция для получения фич изображения
def get_image_features(image_path):
    inputs = load_and_process_image(image_path)
    with torch.no_grad():
        image_features = model(inputs)
    return image_features

def classify_image(image_path):
    image_features = get_image_features(image_path)
    _, preds = torch.max(image_features, 1) 
    return preds.detach().cpu().item()


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
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import pandas as pd
test_folder = "/kaggle/input/private-test/private_test"
model_name = "zer0int/CLIP-GmP-ViT-L-14"
clip_bb = CLIPModel.from_pretrained(model_name).vision_model
tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
model = CustomCLIPModel(clip_bb)
processor = CLIPProcessor.from_pretrained(model_name)
model.load_state_dict(torch.load('model.pt', map_location=device))



output_df = pd.DataFrame(columns=["image_name", "label"])

# Классификация изображений
for filename in os.listdir(test_folder):
    image_path = os.path.join(test_folder, filename)
    predicted_class = classify_image(image_path)
    class_number = predicted_class + 1  # Номер класса начинается с 1
    
    output_df = output_df._append({"image_name": filename, "label": class_number}, ignore_index=True)
 
output_file = 'model_submit.csv'
output_df.to_csv(output_file, index=False)