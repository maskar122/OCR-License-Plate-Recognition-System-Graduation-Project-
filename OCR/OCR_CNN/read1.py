import torch
from reed_model import OCR_CNN  # لو class الموديل في نفس السكربت أو ممكن تنقل class لو في ملف تاني

weights_path = r"C:\Users\LAP-STORE\Desktop\Amit\Graduate_project\OCR\EALPR- LP characters dataset\ocr_cnn.pth"

# تعريف الموديل بنفس عدد الفئات اللي اتعلم عليها
num_classes = 26
model = OCR_CNN(num_classes)
model.load_state_dict(torch.load(weights_path, map_location="cpu"))
model.eval()

# معلومات عن الموديل
print(model)
