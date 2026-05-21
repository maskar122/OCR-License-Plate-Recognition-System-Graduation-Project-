import torch
from reed_model import OCR_CNN  # نفس الملف اللي عرفنا فيه class الموديل

weights_path = r"C:\Users\LAP-STORE\Desktop\Amit\Graduate_project\OCR\EALPR- LP characters dataset\ocr_cnn.pth"
onnx_path = r"C:\Users\LAP-STORE\Desktop\Amit\Graduate_project\OCR\EALPR- LP characters dataset\ocr_cnn.onnx"

# تعريف الموديل
num_classes = 26
model = OCR_CNN(num_classes)
model.load_state_dict(torch.load(weights_path, map_location="cpu"))
model.eval()

# Dummy input بنفس شكل الـ input المتوقع
dummy_input = torch.randn(1, 1, 32, 32)

# التحويل لـ ONNX
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=12,        # مناسب للـ ONNX Runtime
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print(f"✅ تم حفظ الموديل كـ ONNX في: {onnx_path}")
