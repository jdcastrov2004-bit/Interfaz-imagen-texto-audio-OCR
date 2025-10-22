import os
import time
import glob
import cv2
import numpy as np
import pytesseract
import base64
import streamlit as st
from PIL import Image
from gtts import gTTS
from googletrans import Translator

st.title("Reconocimiento Óptico de Caracteres")
image = Image.open('inspector.jpg')
st.image(image, width=360)
st.write(
    "Esta actividad te permite extraer texto desde una imagen usando OCR. "
    "Puedes tomar una foto con la cámara o cargar un archivo. "
    "Luego, si lo deseas, traduce el texto resultante y escúchalo en audio."
)

os.makedirs("temp", exist_ok=True)

text = ""

st.subheader("Elige la fuente de la imagen")
cam_ = st.checkbox("Usar cámara")
if cam_:
    img_file_buffer = st.camera_input("Toma una foto")
else:
    img_file_buffer = None

with st.sidebar:
    st.subheader("Procesamiento de imagen")
    filtro = st.radio("Filtro para imagen de cámara", ("Con filtro", "Sin filtro"))
    st.caption("“Con filtro” invierte los colores y puede mejorar el OCR cuando hay fondo oscuro y letras claras.")

bg_image = st.file_uploader("Cargar imagen", type=["png", "jpg", "jpeg"])
if bg_image is not None:
    file_bytes = bg_image.getvalue()
    np_img = np.frombuffer(file_bytes, np.uint8)
    img_cv = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    text = pytesseract.image_to_string(img_rgb)
    st.image(Image.open(bg_image), caption="Imagen cargada", use_container_width=True)
    st.subheader("Texto detectado (imagen cargada):")
    st.write(text)

if img_file_buffer is not None:
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    if filtro == "Con filtro":
        cv2_img = cv2.bitwise_not(cv2_img)
    img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    cam_text = pytesseract.image_to_string(img_rgb)
    st.subheader("Texto detectado (cámara):")
    st.write(cam_text)
    if cam_text.strip():
        text = cam_text

with st.sidebar:
    st.subheader("Traducción y audio")
    translator = Translator()
    in_lang = st.selectbox(
        "Idioma de entrada",
        ("Inglés", "Español", "Bengali", "Coreano", "Mandarín", "Japonés"),
        index=1
    )
    out_lang = st.selectbox(
        "Idioma de salida",
        ("Inglés", "Español", "Bengali", "Coreano", "Mandarín", "Japonés"),
        index=0
    )
    lang_map = {
        "Inglés": "en",
        "Español": "es",
        "Bengali": "bn",
        "Coreano": "ko",
        "Mandarín": "zh-cn",
        "Japonés": "ja",
    }
    input_language = lang_map[in_lang]
    output_language = lang_map[out_lang]

    english_accent = st.selectbox(
        "Acento (aplica sobre todo para inglés)",
        ("Default", "India", "United Kingdom", "United States", "Canada", "Australia", "Ireland", "South Africa"),
        index=0
    )
    tld_map = {
        "Default": "com",
        "India": "co.in",
        "United Kingdom": "co.uk",
        "United States": "com",
        "Canada": "ca",
        "Australia": "com.au",
        "Ireland": "ie",
        "South Africa": "co.za",
    }
    tld = tld_map[english_accent]
    display_output_text = st.checkbox("Mostrar texto traducido", value=True)

def text_to_speech(input_language, output_language, text, tld):
    translation = translator.translate(text, src=input_language, dest=output_language)
    trans_text = translation.text
    tts = gTTS(trans_text, lang=output_language, tld=tld, slow=False)
    file_stub = "".join(c for c in (text.strip()[:32] or "audio") if c.isalnum() or c in ("-", "_", " ")).strip().replace(" ", "_") or "audio"
    file_path = f"temp/{file_stub}.mp3"
    tts.save(file_path)
    return file_stub, trans_text, file_path

st.subheader("Convertir texto a audio")
if st.button("Traducir y convertir a audio"):
    if not (text or "").strip():
        st.warning("Primero captura o carga una imagen con texto.")
    else:
        try:
            file_stub, output_text, file_path = text_to_speech(input_language, output_language, text, tld)
            with open(file_path, "rb") as f:
                audio_bytes = f.read()
            st.success("Audio generado.")
            st.audio(audio_bytes, format="audio/mp3")
            st.download_button("Descargar MP3", data=audio_bytes, file_name=f"{file_stub}.mp3")
            if display_output_text:
                st.markdown("#### Texto traducido:")
                st.write(output_text)
        except Exception as e:
            st.error(f"Ocurrió un error al generar el audio: {e}")

def remove_files(n_days: int):
    mp3_files = glob.glob("temp/*.mp3")
    if not mp3_files:
        return
    now = time.time()
    horizon = now - n_days * 86400
    for f in mp3_files:
        try:
            if os.stat(f).st_mtime < horizon:
                os.remove(f)
        except Exception:
            pass

remove_files(7)
