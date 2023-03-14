import streamlit as st
from PIL import Image
import model

# Set title
st.title('Birds Birds Birds')

# File uploader widget
upload_image = st.file_uploader("Upload an image", type=['png', 'jpg'])

if upload_image is not None:
    image = Image.open(upload_image)
    image = image.convert("RGB")

    st.image(image, caption='Your Image', use_column_width=True)

    predictions = model.predict(image)

    st.write("Top 3 Predictions (name, score)")
    for i in predictions:
        st.write(i[0], ", ", i[1])