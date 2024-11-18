import torch
from torchvision import transforms, models
from PIL import Image
import os


class Detect:
    def __init__(self, labels_file, model_path, device=None):
        """
        Initialize the Detect class.

        Args:
            labels_file (str): Path to the labels.txt file.
            model_path (str): Path to the trained model file.
            device (str, optional): Device to run the model on. Defaults to CUDA if available.
        """
        self.labels_file = labels_file
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        print(f"Using device: {self.device}")

        # Load class-to-index mapping
        self.idx_to_class = self._load_labels()
        self.num_classes = len(self.idx_to_class)
        print(f"Loaded {self.num_classes} classes from {labels_file}.")

        # Initialize model
        self.model = self._initialize_model()
        self.data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _load_labels(self):
        """
        Load class-to-index mapping from a labels.txt file.
        """
        idx_to_class = {}
        if not os.path.exists(self.labels_file):
            raise FileNotFoundError(f"Labels file does not exist: {self.labels_file}")

        with open(self.labels_file, 'r') as file:
            for idx, line in enumerate(file):
                idx_to_class[idx] = line.strip()
        return idx_to_class

    def _initialize_model(self):
        """
        Load the pre-trained model with the correct classifier layer.
        """
        model = models.mobilenet_v2(pretrained=False)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, self.num_classes)
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        return model

    def _load_image(self, image_path):
        """
        Load and preprocess the image.
        """
        image = Image.open(image_path).convert("RGB")
        return self.data_transforms(image).unsqueeze(0)  # Add batch dimension

    def detect_class(self, image_path):
        """
        Predict the class of the given image.

        Args:
            image_path (str): Path to the test image.

        Returns:
            str: Predicted class name.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image path does not exist: {image_path}")

        image = self._load_image(image_path).to(self.device)
        with torch.no_grad():
            outputs = self.model(image)
            _, predicted_idx = torch.max(outputs, 1)
            predicted_class = self.idx_to_class[predicted_idx.item()]
        return predicted_class


# Example usage
# labels_file = "labels/labels.txt"  # Path to labels file
# model_path = "model/mobilenet_doodle_model.pth"
# test_image_path1 = "test_images/1.png"
# test_image_path2 = "test_images/2.png"
# test_image_path3 = "test_images/3.png"
# test_image_path4 = "test_images/4.png"
#
# test_images = [test_image_path1, test_image_path2, test_image_path3, test_image_path4]
#
# detector = Detect(labels_file=labels_file, model_path=model_path)
#
# # Predict the class of the test image
# try:
#     for f in test_images:
#         predicted_class = detector.detect_class(f)
#         print(f"Predicted Class for {f}: {predicted_class}")
# except FileNotFoundError as e:
#     print(e)