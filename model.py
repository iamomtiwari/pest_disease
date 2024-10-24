import torch
from torchvision import transforms
from PIL import Image

class CropDiseaseModel:
    def __init__(self):
        # Load the model on a CPU-only machine
        self.model = torch.load(r"G:\crop_disease_prediction\model\vit_model.pth", map_location=torch.device('cpu'))

        # Ensure the model is in evaluation mode
        self.model.eval()

        # Define image preprocessing (resize to 224x224, which is required by the ViT model)
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize image to match model input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Standard normalization
        ])

        # Define class names and their solutions
        self.class_names = [
            "Corn___Common_Rust", "Corn___Gray_Leaf_Spot", "Corn___Healthy", "Corn___Northern_Leaf_Blight",
            "Rice___Brown_Spot", "Rice___Healthy", "Rice___Leaf_Blast", "Rice___Neck_Blast",
            "Sugarcane_Bacterial Blight", "Sugarcane_Healthy", "Sugarcane_Red Rot",
            "Wheat___Brown_Rust", "Wheat___Healthy", "Wheat___Yellow_Rust"
        ]
        
        self.solutions = {
            "Corn___Common_Rust": "Apply fungicides as soon as symptoms are noticed. Practice crop rotation and remove infected plants.",
            "Corn___Gray_Leaf_Spot": "Rotate crops to non-host plants, apply resistant varieties, and use fungicides as needed.",
            "Corn___Healthy": "Continue good agricultural practices: ensure proper irrigation, nutrient supply, and monitor for pests.",
            "Corn___Northern_Leaf_Blight": "Remove and destroy infected plant debris, apply fungicides, and rotate crops.",
            "Rice___Brown_Spot": "Use resistant varieties, improve field drainage, and apply fungicides if necessary.",
            "Rice___Healthy": "Maintain proper irrigation, fertilization, and pest control measures.",
            "Rice___Leaf_Blast": "Use resistant varieties, apply fungicides during high-risk periods, and practice good field management.",
            "Rice___Neck_Blast": "Plant resistant varieties, improve nutrient management, and apply fungicides if symptoms appear.",
            "Wheat___Brown_Rust": "Apply fungicides and practice crop rotation with non-host crops.",
            "Wheat___Healthy": "Continue with good management practices, including proper fertilization and weed control.",
            "Wheat___Yellow_Rust": "Use resistant varieties, apply fungicides, and rotate crops.",
            "Sugarcane__Red_Rot": "Plant resistant varieties and ensure good drainage.",
            "Sugarcane__Healthy": "Maintain healthy soil conditions and proper irrigation.",
            "Sugarcane__Bacterial Blight": "Use disease-free planting material, practice crop rotation, and destroy infected plants."
        }

    def predict(self, image_path):
        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        image = self.preprocess(image).unsqueeze(0)  # Preprocess and add batch dimension

        with torch.no_grad():
            output = self.model(image)  # Make prediction
        _, predicted_idx = torch.max(output, 1)  # Get predicted class index

        predicted_class = self.class_names[predicted_idx.item()]  # Map index to class name
        solution = self.solutions.get(predicted_class, "No solution available.")  # Get solution based on class name

        return predicted_class, solution  # Return both the class name and solution
model = CropDiseaseModel()
predicted_class, solution = model.predict("path_to_image.jpg")
print(f"Predicted Class: {predicted_class}")
print(f"Suggested Solution: {solution}")
