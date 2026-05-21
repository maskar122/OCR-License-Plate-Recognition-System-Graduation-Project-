import os
import shutil
import subprocess

# -------------------------
# المسارات
# -------------------------
source_root = r"C:\Users\LAP-STORE\Desktop\Amit\Graduate_project\Database\New folder (2)"
output_root = r"C:\Users\LAP-STORE\Desktop\Amit\Graduate_project\prepare_data2\output_face_detection"
weights_path = r"C:\Users\LAP-STORE\Desktop\Amit\Graduate_project\detection\face\yolov5\runs\train\person_yolo_cpu_416\weights\best.pt"
img_size = 416
conf_thres = 0.3
device = 'cpu'

detect_script = r"C:/Users/LAP-STORE/Desktop/Amit/New folder/yolov5/detect.py"  # مسار detect.py عندك

os.makedirs(output_root, exist_ok=True)

for folder in os.listdir(source_root):
    folder_path = os.path.join(source_root, folder)
    if not os.path.isdir(folder_path):
        continue

    img_path = os.path.join(folder_path, "in_car.jpg")
    if not os.path.exists(img_path):
        print(f"⚠️ الصورة in_car.jpg مش موجودة في {folder}")
        continue

    output_folder = os.path.join(output_root, folder)
    os.makedirs(output_folder, exist_ok=True)

    # -------------------------
    # أمر تشغيل detect.py
    # -------------------------
    cmd = [
        "python", detect_script,
        "--weights", weights_path,
        "--img", str(img_size),
        "--conf", str(conf_thres),
        "--source", img_path,
        "--device", device,
        "--project", output_folder,
        "--name", "",       # نخلي YOLO يحط الصورة مباشرة هنا
        "--exist-ok"
    ]

    subprocess.run(cmd)

    # YOLO بيحفظ الصورة في فولدر exp/
    exp_folder = os.path.join(output_folder, 'exp')
    if os.path.exists(exp_folder):
        for f in os.listdir(exp_folder):
            if f.endswith(".jpg"):
                shutil.move(os.path.join(exp_folder, f), os.path.join(output_folder, "in_car.jpg"))
        shutil.rmtree(exp_folder)

    print(f"✅ تم معالجة {img_path} وحفظها في {output_folder}/in_car.jpg")
