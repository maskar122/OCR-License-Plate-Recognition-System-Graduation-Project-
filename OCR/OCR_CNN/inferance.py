import torch
from torchvision import transforms
from PIL import Image
import os

# -----------------------------
# المسارات
# -----------------------------
model_path = r"C:\Users\LAP-STORE\Desktop\Amit\Graduate_project\OCR\EALPR- LP characters dataset\ocr_cnn.pth"
image_path = r"C:\Users\LAP-STORE\Desktop\Amit\Graduate_project\OCR\EALPR- LP characters dataset\data_split\test\ص\0402_license_plate_1-ص-0.png"  # ضع هنا مسار الصورة

# -----------------------------
# إعداد الـ Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# نفس Transform زي التدريب
# -----------------------------
img_size = 32
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# -----------------------------
# CNN Architecture نفسها
# -----------------------------
class OCR_CNN(torch.nn.Module):
    def __init__(self, num_classes):
        super(OCR_CNN, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2,2),

            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2,2),

            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2,2)
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(128*4*4, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128, 26)  # عدد الـ classes
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# -----------------------------
# Load Model
# -----------------------------
model = OCR_CNN(num_classes=26).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# -----------------------------
# Load Image
# -----------------------------
img = Image.open(image_path).convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(device)  # أضف batch dimension

# -----------------------------
# Inference
# -----------------------------
with torch.no_grad():
    output = model(img_tensor)
    _, pred = torch.max(output, 1)

# -----------------------------
# Print Prediction
# -----------------------------
classes = ['أ', 'ب', 'ج', 'د', 'ر', 'س', 'ص', 'ط', 'ع', 'ف', 'ق', 'ل', 'م', 'ن', 'و', 'ى', '١', '٢', '٣', '٤', '٥', '٦', '٧', '٨', '٩', 'ھ']
predicted_char = classes[pred.item()]
print("الحرف المتوقع:", predicted_char)
