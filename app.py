# Save this as app.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Seed Detection Pipeline")

def process_seeds(image_array, area_threshold=200):
    img = cv2.cvtColor(np.array(image_array), cv2.COLOR_RGB2BGR)
    orig = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21, 1
    )

    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = orig.copy()
    seed_id = 0
    data = []

    for c in cnts:
        area = cv2.contourArea(c)
        if area < area_threshold:
            continue
        x, y, w, h = cv2.boundingRect(c)
        seed_id += 1
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [c], -1, 255, -1)
        mean_val = cv2.mean(orig, mask=mask)[:3]

        brightness = int(np.mean(mean_val))
        if brightness < 85:
            color_label = 'dark'
        elif brightness < 170:
            color_label = 'medium'
        else:
            color_label = 'light'

        data.append({
            'id': seed_id,
            'size_px': area,
            'color': color_label
        })

        cv2.rectangle(output, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(output, str(seed_id), (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    return orig_rgb, output_rgb, seed_id, data

# Upload image
uploaded_file = st.file_uploader("Upload a seed image", type=["png","jpg","jpeg"])
area_thresh = st.slider("Minimum seed area (px)", 50, 1000, 200)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    orig_img, detected_img, seed_count, seed_data = process_seeds(image, area_thresh)

    st.subheader("Original Image")
    st.image(orig_img)

    st.subheader("Detected Grains")
    st.image(detected_img)

    st.subheader("Summary")
    st.text(f"Total seeds detected: {seed_count}")
    for d in seed_data:
        st.text(f"ID:{d['id']}, Size:{d['size_px']}, Color:{d['color']}")
