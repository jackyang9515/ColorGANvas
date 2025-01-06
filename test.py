import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
from model import Unet
import matplotlib.pyplot as plt
"""
This File will be used for manual inference of the model, 1 image at a time
"""

model_path = "generator_final.pth"
# replace with the path of the image you want to run inference on
input_image_path = "colorization/training_small/groom/189.jpg"
output_image_path = "modeltest.png"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = Unet(input_nc=1, output_nc=2, num_downs=7, ngf=64)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

def preprocess_image(image_path):
    """
    Preprocess image to exactly match training preprocessing
    Returns 4D tensor [batch, channel, height, width]
    """
    # Read as RGB and resize to match training
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    image = cv2.resize(image, (256, 256))
    
    # Convert RGB to LAB
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)
    
    # Normalize L to [0, 1] and create 4D tensor [batch, channel, height, width]
    L_tensor = torch.from_numpy(L).float() / 100.0
    L_tensor = L_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    
    return L_tensor, L

def postprocess_output(L_channel, AB_tensor):
    """
    Convert model output back to RGB image, matching training normalization
    """
    # Denormalize AB prediction (was normalized to [-1, 1] during training)
    AB = AB_tensor.squeeze().detach().cpu().numpy() * 128.0 + 128.0
    AB = AB.transpose(1, 2, 0)
    
    # Create LAB image (L channel is already in correct range [0, 100])
    LAB = np.zeros((L_channel.shape[0], L_channel.shape[1], 3), dtype=np.uint8)
    LAB[:, :, 0] = L_channel
    LAB[:, :, 1:] = AB.astype(np.uint8)
    
    # Convert to RGB
    RGB = cv2.cvtColor(LAB, cv2.COLOR_LAB2RGB)
    return RGB

with torch.no_grad():
    # Preprocess input image
    L_tensor, original_L = preprocess_image(input_image_path)
    L_tensor = L_tensor.to(device)
    
    # Forward pass through the model
    AB_tensor = model(L_tensor)
    
    colorized_image = postprocess_output(original_L, AB_tensor[0])
    
    grayscale_image = cv2.cvtColor(original_L[..., np.newaxis], cv2.COLOR_GRAY2RGB)
    
    side_by_side_image = np.hstack((grayscale_image, colorized_image))
    
    # Save the side-by-side image
    cv2.imwrite(output_image_path, cv2.cvtColor(side_by_side_image, cv2.COLOR_RGB2BGR))
    print(f"Side-by-side image saved to {output_image_path}")