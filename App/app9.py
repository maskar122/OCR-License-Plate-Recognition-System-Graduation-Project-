import streamlit as st
import cv2
import numpy as np
import torch
import pickle
import tempfile
import os
from insightface.app import FaceAnalysis

# ==========================================
# CONFIG
# ==========================================
FACE_YOLO_WEIGHTS = r"C:\Users\LAP-STORE\Desktop\Amit\Graduate_project\detection\face\yolov5\runs\train\person_yolo_cpu_416\weights\best.pt"
PLATE_YOLO_WEIGHTS = r"C:\Users\LAP-STORE\Desktop\Amit\New folder\yolov5\runs\train\license_plate_cpu2\weights\best.pt"

DB_EMB_PATH = r"C:\Users\LAP-STORE\Desktop\Amit\Graduate_project\prepare_data\db_embeddings.pkl"
DB_NAME_PATH = r"C:\Users\LAP-STORE\Desktop\Amit\Graduate_project\prepare_data\db_names.pkl"

SIM_THRESHOLD = 0.5

# ==========================================
# LOAD MODELS (ONCE)
# ==========================================
@st.cache_resource
def load_face_yolo():
    return torch.hub.load(
        'ultralytics/yolov5',
        'custom',
        path=FACE_YOLO_WEIGHTS,
        force_reload=False
    )

@st.cache_resource
def load_plate_yolo():
    return torch.hub.load(
        'ultralytics/yolov5',
        'custom',
        path=PLATE_YOLO_WEIGHTS,
        force_reload=False
    )

@st.cache_resource
def load_face_recognizer():
    app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

@st.cache_resource
def load_database():
    with open(DB_EMB_PATH, "rb") as f:
        emb = pickle.load(f)
    with open(DB_NAME_PATH, "rb") as f:
        names = pickle.load(f)
    return emb, names


face_yolo = load_face_yolo()
plate_yolo = load_plate_yolo()
face_app = load_face_recognizer()
db_embeddings, db_names = load_database()

# ==========================================
# UTILS
# ==========================================
def read_image(file):
    data = np.frombuffer(file.read(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

# ==========================================
# UI
# ==========================================
st.set_page_config(page_title="Graduation Project", layout="centered")
st.title("🚗 Digital Driver Verified System")

uploaded_file = st.file_uploader("📤 Upload Car Image", type=["jpg", "png"])

if uploaded_file:
    image = read_image(uploaded_file)
    st.image(image, channels="BGR", caption="Original Image")

    # =============================
    # FACE DETECTION
    # =============================
    if st.button("🔍 Face Detection"):
        results = face_yolo(image)
        detected_img = results.render()[0]
        st.session_state["face_detected"] = image
        st.image(detected_img, channels="BGR", caption="Face Detected")

    # =============================
    # FACE RECOGNITION
    # =============================
    if st.button("🧠 Face Recognition"):
        faces = face_app.get(image)

        if len(faces) == 0:
            st.error("❌ No face detected")
        else:
            face = faces[0]
            emb = face.embedding
            emb = emb / np.linalg.norm(emb)

            sims = np.dot(db_embeddings, emb)
            idx = np.argmax(sims)

            score = float(sims[idx])
            name = db_names[idx] if score >= SIM_THRESHOLD else "Unknown"

            st.success(f"👤 Name: {name}")
            st.info(f"📊 Similarity: {score:.3f}")

    # =============================
    # PLATE DETECTION
    # =============================
    if st.button("🔢 Plate Detection"):
        results = plate_yolo(image)
        plate_img = results.render()[0]
        st.image(plate_img, channels="BGR", caption="Plate Detected")

# =============================
# SHOW ALL RESULTS
# =============================
from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display

if st.button("🎯 Show All Results"):
    if uploaded_file is None:
        st.warning("⚠️ ارفع صورة أولًا")
    else:
        final_img = image.copy()

        # Face Detection + YOLO render
        face_results = face_yolo(final_img)
        final_img = face_results.render()[0]

        # PIL لرسم الاسم العربي
        img_pil = Image.fromarray(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        font_path = "arial.ttf"  # فونت عربي موجود على جهازك
        font = ImageFont.truetype(font_path, 30)

        # إعداد ألوان النص
        text_color = (255,0,0)      # أخضر مثلاً، ممكن تغيره
        stroke_color = (0, 0, 0)      # الإطار الخارجي أسود
        stroke_width = 2               # سمك الإطار

        for box in face_results.xyxy[0].cpu().numpy():  # YOLO boxes
            x1, y1, x2, y2, conf, cls = box.astype(int)

            face_crop = final_img[y1:y2, x1:x2]
            faces = face_app.get(face_crop)

            if len(faces) > 0:
                face = faces[0]
                emb = face.embedding / np.linalg.norm(face.embedding)
                sims = np.dot(db_embeddings, emb)
                idx = np.argmax(sims)
                score = float(sims[idx])
                name = db_names[idx] if score >= SIM_THRESHOLD else "Unknown"
            else:
                name = "Unknown"

            # ترتيب الاسم العربي صح
            reshaped_text = arabic_reshaper.reshape(name)
            bidi_text = get_display(reshaped_text)

            # مكان النص بجانب المربع
            text_margin = 5
            text_x = x2 + text_margin
            text_y = y1

            # رسم الاسم مع stroke لإظهار النص بوضوح
            draw.text(
                (text_x, text_y),
                bidi_text,
                font=font,
                fill=text_color,
                stroke_width=stroke_width,
                stroke_fill=stroke_color
            )

        # رجع الصورة لـ OpenCV
        final_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        # Plate Detection
        plate_results = plate_yolo(final_img)
        final_img = plate_results.render()[0]

        st.image(final_img, channels="BGR", caption="Final Result: Face + Plate Detection + Recognition")

# ==========================================
# NEW CONFIG FOR CHARACTERS
# ==========================================
CHAR_YOLO_WEIGHTS = r"C:\Users\LAP-STORE\Desktop\Amit\Graduate_project\OCR\EALPR- LP characters dataset\pipeline\letter_yolo5\yolov5\runs\train\exp14\weights\best.pt"
CHAR_CNN_WEIGHTS = r"C:\Users\LAP-STORE\Desktop\Amit\Graduate_project\OCR\EALPR- LP characters dataset\ocr_cnn.pth"

# ده نفس اللي اتدرب عليه
CHAR_CLASSES = ['أ','ب','ج','د','ر','س','ص','ط','ع','ف','ق','ل','م','ن','و','ى',
                '١','٢','٣','٤','٥','٦','٧','٨','٩','ھ']

DEVICE = "cpu"

import torch
import torch.nn as nn
from torchvision import transforms

# ==========================================
# LOAD CHARACTERS MODELS
# ==========================================
@st.cache_resource
def load_char_yolo():
    return torch.hub.load(
        'ultralytics/yolov5',
        'custom',
        path=CHAR_YOLO_WEIGHTS,
        force_reload=False
    )

@st.cache_resource
def load_char_cnn():
    # CNN مطابق للتدريب الأصلي
    class OCR_CNN(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2,2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2,2),
                nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2,2)
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128*4*4, 128), nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, num_classes)
            )
        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    model = OCR_CNN(num_classes=len(CHAR_CLASSES))
    model.load_state_dict(torch.load(CHAR_CNN_WEIGHTS, map_location=DEVICE))
    model.eval()
    return model

char_yolo = load_char_yolo()
char_cnn = load_char_cnn()

# ==========================================
# CHAR OCR UTILS
# ==========================================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((32,32)),  # مطابق للتدريب
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def recognize_character(crop_img):
    img_tensor = transform(crop_img).unsqueeze(0)  # add batch dimension
    with torch.no_grad():
        output = char_cnn(img_tensor)
        pred = output.argmax(dim=1).item()
        return CHAR_CLASSES[pred]

# ==========================================
# STREAMLIT BUTTON FOR PLATE OCR (Spaces + RTL)
# ==========================================
import arabic_reshaper
from bidi.algorithm import get_display

if st.button("🔡 Plate OCR"):
    if uploaded_file is None:
        st.warning("⚠️ ارفع صورة أولًا")
    else:
        # 1️⃣ Plate Detection
        plate_results = plate_yolo(image)
        if len(plate_results.xyxy[0]) == 0:
            st.error("❌ No plate detected")
        else:
            x1, y1, x2, y2, conf, cls = plate_results.xyxy[0][0].cpu().numpy().astype(int)
            plate_crop = image[y1:y2, x1:x2]
            st.image(plate_crop, channels="BGR", caption="Cropped Plate")

            # 2️⃣ Characters YOLO Detection
            char_results = char_yolo(plate_crop)
            boxes = char_results.xyxy[0].cpu().numpy().astype(int)

            # ✅ فلترة الـ boxes المتقاربة جداً (avoid duplicate detection)
            boxes = sorted(boxes, key=lambda b: b[0])  # sort by x (left → right)
            filtered_boxes = []
            for b in boxes:
                if len(filtered_boxes) == 0:
                    filtered_boxes.append(b)
                else:
                    if abs(b[0] - filtered_boxes[-1][0]) > 5:
                        filtered_boxes.append(b)
            boxes = filtered_boxes

            # 3️⃣ Crop each character and recognize (بدون إزالة التكرار)
            chars_detected = []
            for box in boxes:
                cx1, cy1, cx2, cy2, conf, cls = box
                char_crop = plate_crop[cy1:cy2, cx1:cx2]
                char_label = recognize_character(char_crop)
                chars_detected.append((cx1, char_label))

            # 4️⃣ ترتيب الحروف حسب موضعها على الصورة (من اليسار لليمين)
            chars_sorted = sorted(chars_detected, key=lambda x: x[0])

            # 5️⃣ فصل الحروف العربية عن الأرقام
            arabic_chars = [c[1] for c in chars_sorted if not c[1].isdigit()]
            numbers = [c[1] for c in chars_sorted if c[1].isdigit()]

            # 6️⃣ ترتيب الحروف العربية من اليمين لليسار مع مسافات
            arabic_chars_text = get_display(arabic_reshaper.reshape(" ".join(arabic_chars)))

            # 7️⃣ دمج العربية + الأرقام بحيث تظهر بشكل صحيح مع مسافة
            plate_text = f"{arabic_chars_text} {' '.join(numbers[::-1])}"

            # 8️⃣ حفظ النص في session_state لاستخدامه في Show Final Results
            st.session_state["plate_text"] = plate_text

            st.success(f"📄 Plate OCR Result: {plate_text}")

# ==========================================
# Show Final Results (Face + Plate + OCR from Plate OCR)
# ==========================================
import arabic_reshaper
from bidi.algorithm import get_display

if st.button("🎯 Show Final Results"):
    if uploaded_file is None:
        st.warning("⚠️ ارفع صورة أولًا")
    else:
        final_img = image.copy()

        # 1️⃣ Face Detection + Recognition
        face_results = face_yolo(final_img)
        final_img = face_results.render()[0]

        # PIL للرسم
        img_pil = Image.fromarray(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        font_path = "arial.ttf"
        font = ImageFont.truetype(font_path, 30)
        text_color = (255,0,0)       # أحمر للأسماء و OCR
        stroke_color = (0,0,0)
        stroke_width = 2

        # رسم أسماء الأشخاص
        for box in face_results.xyxy[0].cpu().numpy():
            x1, y1, x2, y2, conf, cls = box.astype(int)
            face_crop = final_img[y1:y2, x1:x2]
            faces = face_app.get(face_crop)
            if len(faces) > 0:
                face = faces[0]
                emb = face.embedding / np.linalg.norm(face.embedding)
                sims = np.dot(db_embeddings, emb)
                idx = np.argmax(sims)
                score = float(sims[idx])
                name = db_names[idx] if score >= SIM_THRESHOLD else "Unknown"
            else:
                name = "Unknown"
            reshaped_text = arabic_reshaper.reshape(name)
            bidi_text = get_display(reshaped_text)
            draw.text((x2+5, y1), bidi_text, font=font, fill=text_color,
                      stroke_width=stroke_width, stroke_fill=stroke_color)

        final_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        # 2️⃣ Plate Detection
        plate_results = plate_yolo(final_img)
        final_img = plate_results.render()[0]

        # 3️⃣ Plate OCR Text جنب اللوحة
        if "plate_text" in st.session_state and len(plate_results.xyxy[0]) > 0:
            x1, y1, x2, y2, conf, cls = plate_results.xyxy[0][0].cpu().numpy().astype(int)
            plate_text = st.session_state["plate_text"]

            # إعادة ترتيب النص للـ RTL قبل رسمه
            reshaped_ocr = arabic_reshaper.reshape(plate_text)
            bidi_ocr = get_display(reshaped_ocr)

            img_pil = Image.fromarray(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            draw.text((x2+5, y1), bidi_ocr, font=font, fill=text_color,
                      stroke_width=stroke_width, stroke_fill=stroke_color)
            final_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        st.image(final_img, channels="BGR", caption="Final Result: Face + Plate + OCR")
