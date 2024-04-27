import torch
from torchvision import models, transforms
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np

# Initialize MTCNN for face detection
detector = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')

# Load a pretrained ResNet model using the new weights system
model = InceptionResnetV1(pretrained='vggface2').eval()
model.eval()  # Set the model to inference mode
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Transform pipeline for preprocessing the images
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_face_embedding(image_path):
    """Extracts face embedding from the image."""
    image = Image.open(image_path)
    image = image.convert('RGB')
    
    # Detect faces in the image
    boxes, _ = detector.detect(image)
    if boxes is not None:
        # Assuming the first detected face is the one we want
        x, y, width, height = boxes[0]
        face_image = image.crop((x, y, x + width, y + height))
        face_image = transform(face_image).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')
        # print(f'FACE IMAGE {face_image}')
        # Use ResNet to extract features
        with torch.no_grad():
            embedding = model(face_image)
        return embedding
    return None

# Paths to your images
img1_path = "C:\\Users\\0xtriboulet\\OneDrive\\Desktop\\maldev\\Zz.Projects\\RustyVision\\RustyEye\\img\\obama_1.png"
img2_path = "C:\\Users\\0xtriboulet\\OneDrive\\Desktop\\maldev\\Zz.Projects\\RustyVision\\RustyEye\\img\\obama_3.png"

# Get embeddings
embedding1 = get_face_embedding(img1_path)
embedding2 = get_face_embedding(img2_path)

if embedding1 is not None and embedding2 is not None:
    # Calculate cosine similarity
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    similarity = cos(embedding1, embedding2)
    print(f"Similarity: {similarity.item()}")
else:
    print("Face not detected in one or both images.")

# Convert model to TorchScript and save
example_input = torch.rand(1, 3, 256, 256).to('cuda' if torch.cuda.is_available() else 'cpu')
traced_model = torch.jit.trace(model, example_input)
traced_model.save("model.pt")
print("Model has been converted to TorchScript and saved as model.pt")
