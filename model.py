import torch
from torchvision import transforms
from PIL import Image

class CropDiseaseModel:
    def __init__(self):
        # Load the trained model
        self.model = torch.load(r"C:\Users\Hp\Downloads\vit_model.pth")
        self.model.eval()

        # Define image preprocessing (resize to 224x224, which is required by the ViT model)
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize image to match model input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Standard normalization
        ])

        # Define class names and their solutions
        self.class_names = [
            "Wheat Leaf Rust", 
            "Wheat Yellow Rust", 
            "Wheat Septoria", 
            "Rice Brown Spot", 
            "Rice Bacterial Leaf Blight", 
            "Rice Leaf Blast"
        ]
        self.solutions = {
            "Wheat Leaf Rust": "Use fungicides like triadimefon and reduce nitrogen fertilizer.",
            "Wheat Yellow Rust": "Apply resistant cultivars or fungicides.",
            "Wheat Septoria": "Reduce plant density, use fungicides like azoxystrobin.",
            "Rice Brown Spot": "Increase potassium fertilizers, and apply fungicides like mancozeb.",
            "Rice Bacterial Leaf Blight": "Apply copper-based bactericides and use resistant varieties.",
            "Rice Leaf Blast": "Ensure good drainage and apply fungicides such as tricyclazole."
        }

    def predict(self, image_path):
        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        image = self.preprocess(image).unsqueeze(0)  # Preprocess and add batch dimension

        with torch.no_grad():
            output = self.model(image)  # Make prediction
        _, predicted_idx = torch.max(output, 1)  # Get predicted class index

        predicted_class = self.class_names[predicted_idx.item()]  # Map index to class name
        solution = self.solutions[predicted_class]  # Get solution based on class name

        return predicted_class, solution  # Return both the class name and solution
