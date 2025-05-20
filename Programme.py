"""Reconnaissance d'images avec PyTorch et ResNet50 pré-entraîné"""

import torch
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image

# Charger modèle ResNet50 avec les bons poids
weights = ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)
model.eval()

# Charger les noms des classes depuis les poids
categories = weights.meta["categories"]

# Transformations pour préparer l'image
preprocess = weights.transforms()

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0)  # batch de taille 1

    with torch.no_grad():
        output = model(input_tensor)

    # Probabilités softmax
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Top 5 résultats
    top5_prob, top5_catid = torch.topk(probabilities, 5)

    print("\nRésultat de la reconnaissance d'image :")
    for i in range(top5_prob.size(0)):
        print(f"{categories[top5_catid[i]]}: {top5_prob[i].item()*100:.2f}%")

if __name__ == "__main__":
    image_path = r"C:\Users\zakar\IA\Picture-Ai\Bateau.png"  # REMPLACEZ ici 
    predict_image(image_path)
