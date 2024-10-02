import torch
import torchvision
from torchvision import transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_FOLDER = "../Dataset/imagenet-mini/train"

# Can be changed to any other class
CLASS_NAME = "340" # Zebra

# This is a part of preprocessing step, where we remove the class name from the generated keyword,
# Because the class is always an explanation for the class.
REMOVE_WORD = "zebra" # Remove this word from the generated keyword. 

# Keywords config
NUM_PATCHES = 3
QA_MODEL = "Salesforce/blip-vqa-capfilt-large" # QA model name

# Questions to ask about the image
# Can be changed to any other questions of your choice
QUESTIONS = ["What is the pattern in the image?",
             "What are the colors in the image?", 
             "What is the background color of the image?",
             "What is in the background of the image?",
             "What is the primary texture in the image?",
             "What is the secondary texture in the image?",
             "What is the shape of the image?"]

# TCAV config
TRANSFORMS = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ])

# Change it to wantever layer you want to analyse.
LAYERS = ['inception4e']
CLASS_TO_BE_EXPLAINED = 340 # Zebra (Class number)
RANDOM_FOLDER_NAME = "random concept" # Random concept folder name
MODEL_TO_EXPLAIN = torchvision.models.googlenet(weights=torchvision.models.GoogLeNet_Weights.IMAGENET1K_V1) # Model to explain (NUT)
MODEL_TO_EXPLAIN = MODEL_TO_EXPLAIN.eval()

# Diffusion config
N_SAMPLES = 10 # Number of samples to generate
GEN_MODEL_NAME = "runwayml/stable-diffusion-v1-5" # Generative Model name
DPO_WEIGHTS_PATH = "./" # Path to save the DPO weights
DPO_WEIGHTS_NAME = "lora_dpo_weights.safetensors" # DPO weights name
GEN_IMAGE_SIZE = 512 # Image size to generated image
NUM_INFERENCE_STEPS = 25 # Number of inference steps
GRADIENT_STEP_ACCUMULATION = 20 # Gradient accumulation steps

# RL config
TOTAL_RL_STEP = 500 # Total number of RL steps