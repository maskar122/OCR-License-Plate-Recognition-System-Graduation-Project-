import os
import shutil
from sklearn.model_selection import train_test_split
import re

# -----------------------------
# المسارات الأساسية
# -----------------------------
images_dir = r"C:\Users\LAP-STORE\Desktop\Amit\Graduate_project\OCR\EALPR- LP characters dataset\Characters"
output_dir = r"C:\Users\LAP-STORE\Desktop\Amit\Graduate_project\OCR\EALPR- LP characters dataset\data_split"

# -----------------------------
# اقرأ الصور واستخرج الحروف
# -----------------------------
data = []

for img_file in os.listdir(images_dir):
    if img_file.lower().endswith((".png", ".jpg", ".jpeg")):
        # استخرج الحرف من اسم الصورة
        # نفترض الشكل XXX-<char>-Y.png
        match = re.search(r'-(.)-', img_file)
        if match:
            char_label = match.group(1)
            data.append((img_file, char_label))
        else:
            print(f"تعذر استخراج الحرف من اسم الصورة: {img_file}")

print(f"عدد الصور اللي هننظمها: {len(data)}")
print("أمثلة من data:", data[:5])

# -----------------------------
# تقسيم الداتا: Train / Val / Test
# -----------------------------
train_val, test = train_test_split(data, test_size=0.1, random_state=42)
train, val = train_test_split(train_val, test_size=0.2, random_state=42)

print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

# -----------------------------
# دالة لنقل الملفات
# -----------------------------
def move_files(data_list, split_name):
    for img_name, label in data_list:
        src = os.path.join(images_dir, img_name)
        dst_dir = os.path.join(output_dir, split_name, label)
        os.makedirs(dst_dir, exist_ok=True)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(dst_dir, img_name))
        else:
            print(f"الصورة مش موجودة: {src}")

# -----------------------------
# تنفيذ النقل
# -----------------------------
move_files(train, "train")
move_files(val, "val")
move_files(test, "test")

print("تم تنظيم الداتا بنجاح ✅")
