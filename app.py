from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import base64
import io
import os

app = Flask(__name__)
CORS(app)

# Define your CNN model class (must match training architecture)
class CavityCNN(nn.Module):
    def __init__(self):
        super(CavityCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 30 * 30, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)

# Load model
model = CavityCNN()
model.load_state_dict(torch.load('cavity_model.pth', map_location=torch.device('cpu')))
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

@app.route('/')
def index():
    return render_template('index.html')  # index.html must be in templates/

@app.route('/api/detect-cavity', methods=['POST'])
def detect_cavity():
    try:
        data = request.get_json()
        image_data = data['image']
        
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = model(img_tensor)
            pred = torch.round(output).item()
            confidence = output.item()
        
        if pred == 1:
            result = {
                'hasCavity': False,
                'confidence': confidence,
                'details': 'No cavity detected'
            }
        else:
            result = {
                'hasCavity': True,
                'confidence': 1 - confidence,
                'details': 'Cavity detected'
            }

        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
