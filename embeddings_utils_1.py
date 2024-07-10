from transformers import AutoFeatureExtractor, AutoModel
import torch
import torch.nn.functional as F
from torch import Tensor
import os
import warnings
from PIL import Image
import json

# The transformers library internally is creating this warning, but does not
# impact our app. Safe to ignore.
warnings.filterwarnings(action='ignore', category=ResourceWarning)

# We won't have competing threads in this example app
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize feature_extractor and model for resnet-50
feature_extractor = AutoFeatureExtractor.from_pretrained('microsoft/resnet-50')
model = AutoModel.from_pretrained('microsoft/resnet-50')

# Model in evaluation mode
model.eval()

def average_pool(last_hidden_states: Tensor) -> Tensor:
    pooled_output = last_hidden_states.mean(dim=[2, 3])  # Pooling over the spatial dimensions
    assert pooled_output.shape[1] == 2048, f"Expected dimension 2048, but got {pooled_output.shape[1]}"
    return pooled_output

def preprocess_image(image_path):
    """Carga y preprocesa una imagen desde una ruta dada."""
    image = Image.open(image_path).convert('RGB')
    inputs = feature_extractor(images=image, return_tensors='pt', padding=True)
    return inputs['pixel_values']

def generate_image_embeddings(image_paths):
    """Genera embeddings para un batch de imágenes."""
    image_tensors = [preprocess_image(image_path) for image_path in image_paths]
    image_batch = torch.cat(image_tensors, dim=0)
    
    with torch.no_grad():
        outputs = model(pixel_values=image_batch)
    
    embeddings = average_pool(outputs.last_hidden_state)
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    return embeddings.numpy().tolist()

def save_embeddings_to_json(data, output_file):
    """Guarda los embeddings en un archivo JSON."""
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)


def get_image_paths_from_directory(directory):
    """Obtiene todas las rutas de imágenes desde una carpeta general y sus subcarpetas."""
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_paths.append(os.path.join(root, file))
    return image_paths
