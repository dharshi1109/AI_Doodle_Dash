import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
import io
import cv2
from streamlit_drawable_canvas import st_canvas
model_path = r"C:\Users\maha9\Downloads\resnet50_quickdraw_model.h5"
model = tf.keras.models.load_model(model_path)
categories = ['circle', 'square', 'triangle', 'star', 'hexagon']
category_images = {
    'circle': r"C:\Users\maha9\OneDrive\Desktop\circle.png",
    'square': r"C:\Users\maha9\OneDrive\Desktop\square.png",
    'triangle': r"C:\Users\maha9\OneDrive\Desktop\triangle.png",
    'star': r"C:\Users\maha9\OneDrive\Desktop\star.png",
    'hexagon': r"C:\Users\maha9\OneDrive\Desktop\hexagon.png"
}
def preprocess_image(image):
    image = image.convert('L')  
    image = image.resize((32, 32))  
    image = np.array(image) / 255.0  
    image = np.expand_dims(image, axis=-1) 
    image = np.expand_dims(image, axis=0) 
    image = np.repeat(image, 3, axis=-1)  
    return image

def predict_with_model(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    class_idx = np.argmax(prediction[0])
    return categories[class_idx]

def classify_shape(image):
    image_np = np.array(image.convert('L'))
    edges = cv2.Canny(image_np, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "Unknown"
    for contour in contours:
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        num_sides = len(approx)
        if num_sides >= 10:
            hull = cv2.convexHull(approx)
            area = cv2.contourArea(hull)
            perimeter = cv2.arcLength(hull, True)
            roundness = 4 * np.pi * area / (perimeter * perimeter)
            if roundness < 0.5: 
                return "star"
        if num_sides == 3:
            return "triangle"
        elif num_sides == 4:
            x, y, w, h = cv2.boundingRect(approx)
            return "square"
        elif num_sides == 6:
            return "hexagon"
        elif num_sides > 10:
            return "circle"

    return "Unknown"
st.set_page_config(page_title="Quick Draw Classifier", layout="wide")
st.title('🎨 AI DOODLE DASH 🖼️')
st.sidebar.header('🎨 Drawing Guide')
st.sidebar.text('Draw images of the following shapes:')
for category in categories:
    st.sidebar.image(category_images[category], caption=category, use_column_width=True)
st.write("🖍️ Draw an image below (32x32 pixels):")
canvas_result = st_canvas(
    fill_color="white",
    width=280,
    height=280,
    stroke_width=10,
    stroke_color="black",
    background_color="white",
    update_streamlit=True,
    key="canvas"
)
if st.button('🔍 Predict'):
    if canvas_result.image_data is not None: 
        image = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA').convert('RGB')
        model_prediction = predict_with_model(image)
        st.write(f'📝 Model Predicted Category: {model_prediction}')
        shape_prediction = classify_shape(image)
        st.write(f'📝 CV Algorithm Predicted Shape: {shape_prediction}')
        st.write("🎨 Your Drawing:")
        st.image(image, caption='Drawing', use_column_width=True)
    else:
        st.write("✏️ Please draw something first.")
st.markdown("""
    <style>
        .css-18e3th9 { 
            background-color: #fafafa;
            color: #333;
        }
        .css-1d391kg {
            color: #007bff;
        }
        .css-1l8vbmc {
            background-color: #007bff;
            color: #fff;
            border-radius: 5px;
        }
    </style>
    """, unsafe_allow_html=True)
