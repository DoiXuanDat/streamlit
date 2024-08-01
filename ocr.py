import easyocr as ocr
import streamlit as st
from PIL import Image
import numpy as np
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

# title
st.title('OCR Application with EasyOCR and VietOCR')

# image uploader
image = st.file_uploader(label='Upload your image here', type=['png', 'jpg', 'jpeg'])

# model selection
ocr_model = st.selectbox("Choose OCR model", ("EasyOCR", "VietOCR"))


@st.cache_resource
def load_easyocr_model():
    reader = ocr.Reader(['en'], model_storage_directory='.')
    return reader


@st.cache_resource
def load_vietocr_model():
    config = Cfg.load_config_from_name('vgg_transformer')
    config['weights'] = r'D:\python_code\ocr-streamlit\streamlit-ocr\weight\weightvietocr.pth'
    config['device'] = 'cpu'
    detector = Predictor(config)
    return detector


if ocr_model == "EasyOCR":
    reader = load_easyocr_model()
elif ocr_model == "VietOCR":
    detector = load_vietocr_model()

if image is not None:
    input_image = Image.open(image)  # read image
    st.image(input_image)

    with st.spinner('Processing'):
        if ocr_model == "EasyOCR":
            result = reader.readtext(np.array(input_image))
            result_text = [text[1] for text in result]
        elif ocr_model == "VietOCR":
            if input_image.mode != 'RGB':
                input_image = input_image.convert('RGB')  # convert to RGB if it's not
            result = detector.predict(input_image)
            result_text = [result]

        st.write(result_text)
    st.success('Success')
    st.balloons()
else:
    st.write("Upload an Image")
