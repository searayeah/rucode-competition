import os

import torch
from PIL import Image
from tqdm.notebook import tqdm
from transformers import CLIPModel, CLIPProcessor

folder_path = "public_test"
class_labels = [
    "Hotel exterior, outdoor area, or building facade",
    "Hotel room, living space, or bedroom with furniture",
    "Swimming pool or hotel pool area",
    "Billiard table, pool table, or game room",
    "Bathroom with toilet, shower, sink, or bath amenities",
    "Hotel restaurant, dining room, or eating area",
    "Hotel lobby, reception area, or entrance hall",
    "Beachfront, shoreline, or sandy beach area",
    "Corridors, hallways, or staircases in the hotel",
    "Food dishes, meals on plates, or table settings",
    "Conference room, meeting room, or seminar space",
    "Gym, fitness center, or exercise equipment area",
    "Balcony view, outdoor balcony, or terrace",
    "Terrace, patio, or outdoor courtyard",
    "Spa, sauna, wellness center, or relaxation area",
]

model_name = "apple/DFN5B-CLIP-ViT-H-14-378"


# Load the model and processor
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)
model.cuda()
model.eval()


def load_images_from_folder(image_folder):
    image_paths = [
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if f.endswith((".png", ".jpg", ".jpeg"))
    ]
    images = []
    for image_path in tqdm(image_paths):
        image = Image.open(image_path)
        images.append((image, image_path))
    return images


def apply_model_to_images(image_folder, class_labels, batch_size=10):
    images = load_images_from_folder(image_folder)
    all_probs = {}

    with torch.no_grad():
        for i in tqdm(range(0, len(images), batch_size)):
            batch = images[i : i + batch_size]
            batch_images = [img[0] for img in batch]
            batch_image_paths = [img[1] for img in batch]

            # Preprocess the batch of images
            inputs = processor(
                text=class_labels,
                images=batch_images,
                return_tensors="pt",
                padding=True,
            )

            # Move inputs to the appropriate device (CPU or GPU)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Get model outputs
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image  # image-text similarity scores
            probs = logits_per_image.softmax(dim=1)  # probability scores

            # Store probabilities for each image
            for j, image_path in enumerate(batch_image_paths):
                all_probs[image_path] = probs[j].cpu().numpy()

    return all_probs


image_folder = folder_path
batch_size = 10  # Adjust batch size if needed

probs = apply_model_to_images(image_folder, class_labels, batch_size=batch_size)

# # Print out the probabilities
# for img_path, prob in probs.items():
#     print(f"Image: {img_path}, Probabilities: {prob}")
