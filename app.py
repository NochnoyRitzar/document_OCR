import cv2
import streamlit as st
import numpy as np
from paddleocr import PaddleOCR, draw_ocr

ocr = PaddleOCR(use_space_char=True, lang='en')

st.title('Document Text Retrieval and Analysis with PaddleOCR')
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Detecting...")

    # Perform OCR on the image
    result = ocr.ocr(image, cls=True)

    boxes = [res[0] for res in result[0]]
    txts = [res[1][0] for res in result[0]]
    scores = [res[1][1] for res in result[0]]

    # Draw results on the image and display it
    annotated_image = draw_ocr(image, boxes, font_path='PaddleOCR/doc/fonts/latin.ttf')
    st.image(annotated_image, caption='Annotated Image.', use_column_width=True)

    # Display the OCR results
    st.write("Detected Text:")
    for txt in txts:
        st.write(txt)
