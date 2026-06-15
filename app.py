from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "modelo" / "best.pt"
DEFAULT_IOU_THRESHOLD = 0.45


st.set_page_config(
    page_title="Detector de Piletas",
    page_icon="target",
    layout="wide",
)

st.title("Detector de Piletas con YOLO26")
st.markdown("Sube una imagen para detectar piletas con el modelo YOLO.")


@st.cache_resource
def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)


if not MODEL_PATH.exists():
    st.error(f"No se encontro el modelo en: {MODEL_PATH}")
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as e:
    st.error(f"Error cargando modelo: {e}")
    st.stop()

st.sidebar.header("Configuración ⚙️")
confidence = st.sidebar.slider("Confianza minima", 0.0, 1.0, 0.5, 0.05)


def detect_image(image: Image.Image, conf: float, iou_threshold: float):
    image_array = np.array(image.convert("RGB"))
    results = model.predict(source=image_array, conf=conf, iou=iou_threshold, verbose=False)
    result = results[0]
    annotated_frame = result.plot()
    annotated_frame = annotated_frame[:, :, ::-1]
    return annotated_frame, result


st.subheader("Sube tu imagen")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader(
        "Selecciona una imagen",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Imagen original", use_container_width=True)

with col2:
    if uploaded_file is not None:
        with st.spinner("Detectando objetos..."):
            annotated_image, results = detect_image(
                image,
                confidence,
                DEFAULT_IOU_THRESHOLD,
            )

        st.image(
            annotated_image,
            caption="Detecciones",
            use_container_width=True,
        )

        boxes = results.boxes
        if boxes is not None and len(boxes) > 0:
            st.subheader("Resultados de deteccion")
            st.write(f"Total de objetos detectados: **{len(boxes)}**")

            detections_data = []
            for i, box in enumerate(boxes):
                conf = float(box.conf[0].item())
                cls_id = int(box.cls[0].item())
                class_name = model.names.get(cls_id, str(cls_id))
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detections_data.append(
                    {
                        "ID": i + 1,
                        "Clase": class_name,
                        "Confianza": f"{conf:.2%}",
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2),
                    }
                )

            st.table(detections_data)

            st.subheader("Resumen por clase")
            class_summary = {}
            for box in boxes:
                cls_id = int(box.cls[0].item())
                class_name = model.names.get(cls_id, str(cls_id))
                class_summary[class_name] = class_summary.get(class_name, 0) + 1

            for class_name, count in class_summary.items():
                st.write(f"**{class_name}:** {count}")
        else:
            st.warning("No se detectaron objetos en la imagen.")
