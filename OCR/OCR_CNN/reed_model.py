import torch
from train import OCR_CNN  # استيراد class الموديل
import os

# -----------------------------
# مسار weight
# -----------------------------
weights_path = r"C:\Users\LAP-STORE\Desktop\Amit\Graduate_project\OCR\EALPR- LP characters dataset\ocr_cnn.pth"

# -----------------------------
# عدد الفئات (اعتمادًا على بياناتك)
# لو عندك train_dataset.classes تقدر تعمل len(classes)
# -----------------------------
num_classes = 26  # عدد الحروف/الأرقام اللي اتعلم عليها الموديل
  # مثال: 26 حرف + 10 رقم

# -----------------------------
# تعريف الموديل وتحميل الوزن
# -----------------------------
model = OCR_CNN(num_classes)
model.load_state_dict(torch.load(weights_path, map_location="cpu"))
model.eval()  # مهم قبل التحويل

# -----------------------------
# Dummy input للفحص
# -----------------------------
dummy_input = torch.randn(1, 1, 32, 32)  # batch=1, grayscale, 32x32

# -----------------------------
# معلومات للتحويل
# -----------------------------
print("=== معلومات الموديل ===")
print(f"Class الموديل: {model.__class__.__name__}")
print(f"عدد البراميترز القابلة للتعلم: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
print(f"شكل الـ input المتوقع: {dummy_input.shape}")

# تجربة forward لعمل output shape
with torch.no_grad():
    output = model(dummy_input)
print(f"شكل الـ output المتوقع: {output.shape}")

# طباعة هيكل الطبقات
print("\n=== هيكل الموديل ===")
print(model)
