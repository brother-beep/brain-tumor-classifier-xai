import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

# ---------------- Custom Model ---------------- #
class CustomCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # Conv1
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Conv2
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Conv3
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128 * 16 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        self.activations = None
        def save_activation(module, input, output):
            self.activations = output

        self.features[-2].register_forward_hook(save_activation)  # Hook before last MaxPool
        x = self.features(x)
        self.feature_maps = x  # Save for Grad-CAM
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ---------------- Grad-CAM Function ---------------- #
import numpy as np
import cv2
import torch.nn.functional as F
import matplotlib.pyplot as plt

def generate_gradcam(model, input_tensor, target_class):
    gradients = []
    activations = []

    def save_gradient(grad):
        gradients.append(grad)

    # Register hook to the last conv layer
    for name, module in model.features._modules.items():
        if isinstance(module, nn.Conv2d):
            last_conv_layer = module

    def forward_hook(module, input, output):
        activations.append(output)
        output.register_hook(save_gradient)

    handle = last_conv_layer.register_forward_hook(forward_hook)

    # Forward pass
    model.eval()
    output = model(input_tensor)
    model.zero_grad()
    class_loss = output[0, target_class]
    class_loss.backward()

    # Get gradients and activations
    grad = gradients[0].cpu().data.numpy()[0]
    act = activations[0].cpu().data.numpy()[0]

    # Global average pooling of gradients
    weights = np.mean(grad, axis=(1, 2))

    cam = np.zeros(act.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * act[i]

    cam = np.maximum(cam, 0)  # ReLU
    cam = cv2.resize(cam, (128, 128))
    cam -= cam.min()
    cam /= cam.max()

    handle.remove()

    return cam


# ---------------- Setup ---------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomCNN(num_classes=4).to(device)
model.load_state_dict(torch.load("hybrid_model_weights.pth", map_location=device))
model.eval()

class_names = ['giloma', 'meningiloma', 'no_tumor', 'pituitary']

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------------- Streamlit App ---------------- #
st.title("🧠 Brain Tumor Detector")
st.write("Upload an MRI image to predict the tumor class and visualize the reasoning using Grad-CAM.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_tensor = transform(image).unsqueeze(0).to(device)
    img_tensor.requires_grad = True

    with torch.no_grad():
        output = model(img_tensor)
        _, pred = torch.max(output, 1)
        predicted_class = class_names[pred.item()]
    
    st.success(f"🎯 **Predicted tumor: {predicted_class}**")

    # Grad-CAM explanation
    model.zero_grad()
    output = model(img_tensor)
    target_class = pred.item()
    output[0, target_class].backward()
    
    # Save gradients
    model.feature_maps.retain_grad()

    heatmap = generate_gradcam(model, img_tensor, target_class)

    # Convert original image for overlay
    original_img = np.array(image.resize((128, 128)))
    heatmap_img = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_img, 0.5, heatmap_img, 0.5, 0)

    # Plot
    st.markdown("### 🔍 Grad-CAM Explanation")
    st.image(overlay, caption="Highlighted Regions Important for Prediction", use_column_width=True)
    
    # ---------------- Explanation Agent ---------------- #
    def explain_prediction(tumor_class, heatmap):
        explanation = ""

        if tumor_class != "no_tumor":
            explanation += f"🧠 **Tumor Type Detected:** `{tumor_class.capitalize()}`\n\n"

            # Simple interpretation of Grad-CAM
            intensity = heatmap.mean()
            if intensity > 0.5:
                explanation += (
                    "📍 The model focused **strongly** on certain regions of the brain MRI, "
                    "indicating abnormal intensities that are commonly associated with tumors.\n\n"
                )
            else:
                explanation += (
                    "📍 The model identified subtle patterns in the brain tissue that may suggest "
                    "a possible tumor region, though the signal is relatively weak.\n\n"
                )

            # Tumor-specific advice
            if tumor_class == "giloma":
                explanation += (
                    "**Glioma Advice:** Gliomas are aggressive brain tumors. Please consult a neurologist immediately. "
                    "Further imaging (MRI with contrast) and biopsy might be needed."
                )
            elif tumor_class == "meningiloma":
                explanation += (
                    "**Meningioma Advice:** Meningiomas are often benign but may cause pressure on the brain. "
                    "Surgical removal is common. Follow up with a neurosurgeon."
                )
            elif tumor_class == "pituitary":
                explanation += (
                    "**Pituitary Tumor Advice:** These affect hormones and vision. Blood tests and MRI follow-ups are essential. "
                    "Endocrinology consultation is recommended."
                )
        else:
            explanation += (
                "✅ **No tumor detected.**\n\n"
                "The model did not find any abnormal region in the image. "
                "However, if symptoms persist, consult a neurologist for a second opinion."
            )

        return explanation


    # Show Explanation
    st.markdown("### 🤖 Explanation Agent")
    explanation_text = explain_prediction(predicted_class, heatmap)
    st.markdown(explanation_text)


