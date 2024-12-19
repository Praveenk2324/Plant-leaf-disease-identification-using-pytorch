import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnet34
from torchvision.datasets import ImageFolder
import os
import torchvision.models as models
import torch.nn as nn

dataset = ImageFolder(root='archive (2)', transform=transforms.ToTensor()) 
classes = dataset.classes 
data ='archive (2)/'
os.listdir(data)
transform = transforms.Compose([
    transforms.Resize(size=128),
    transforms.ToTensor(),
])
dataset = ImageFolder(data+'New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train',transform=transform)
test_ds = ImageFolder(data+'New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid',transform=transform)
os.listdir(data+'New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)')
os.listdir(data+'New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train')

class Plant_Disease_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = models.resnet34(pretrained=True)
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 38)

    def forward(self, xb):
        out = self.network(xb)
        return out

# Load the trained model
model = Plant_Disease_Model()  # Replace with the appropriate model class
model.load_state_dict(torch.load('plantDisease-resnet34.pth'))
model.eval()

# Define image transformation

# Function to predict the disease
def predict(image):
    img = Image.open(image)
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
    _, predicted = torch.max(output, 1)
    predicted_class = dataset.classes[predicted.item()]  # Assuming you have the 'dataset' and 'classes' defined
    return predicted_class

# Streamlit app
def main():
    st.title("Plant Disease Classifier")
    st.text("Upload an image of a plant leaf.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)
        st.write("")

        if st.button("Predict"):
            prediction = predict(uploaded_file)
            st.success(f"Predicted Disease: {prediction}")

if __name__ == "__main__":
    main()
