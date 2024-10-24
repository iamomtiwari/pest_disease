def __init__(self):
    # Load the model on a CPU-only machine
    self.model = torch.load(r"G:\crop_disease_prediction\model\vit_model.pth", map_location=torch.device('cpu'), weights_only=True)
    # Ensure the model is in evaluation mode
    self.model.eval()
