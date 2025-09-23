import streamlit as st
import cv2
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import os
import pandas as pd
import io
from PIL import Image, ImageDraw, ImageFont
# --- Classification functions ---
def aspect_ratio_classification(ratio):
    ratio = round(ratio, 1)
    if ratio >= 3:
        return "Thin Grain"
    elif 2.1 <= ratio < 3:
        return "Medium Grain"
    elif 1.1 <= ratio < 2.1:
        return "Thick Grain"
    else:
        return "Chalky Grain"
def full_broken_classification(ratio):
    return "Full Sound Grain" if ratio > 1.36 else "Broken Grain"
# --- Color maps ---
aspect_colors = {
    "Thin Grain": "#1F77B4",
    "Medium Grain": "#FF7F0E",
    "Thick Grain": "#D62728",
    "Chalky Grain": "#9467BD",
    "Impurity": "#17BECF",
    "Unknown": "#7F7F7F"
}
# --- Short labels for bar chart ---
short_labels = {
    "Thin Grain": "Thin",
    "Medium Grain": "Medium",
    "Thick Grain": "Thick",
    "Chalky Grain": "Chalky",
    "Impurity": "Impurity",
    "Unknown": "Unknown"
}
# --- Streamlit setup ---
st.set_page_config(layout="wide")
st.title("Rice Grain Analysis")
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)
uploaded_file = st.file_uploader("Upload a rice grain image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    # --- Read image ---
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    # --- Preprocessing ---
    _, binary = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY)
    filtered = cv2.blur(binary, (5,5))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    morph = cv2.dilate(cv2.erode(filtered, kernel), kernel)
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    aspect_counts = Counter()
    # --- Annotated image with Matplotlib ---
    fig, ax = plt.subplots(figsize=(12,12), dpi=300)
    ax.imshow(img, cmap='gray')
    ax.axis('off')
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w)/h if h != 0 else 0
        aspect_ratio = max(aspect_ratio, 1/aspect_ratio)
        aspect_label = aspect_ratio_classification(aspect_ratio)
        aspect_counts[aspect_label] += 1
        rect = plt.Rectangle((x, y), w, h, linewidth=1.5, edgecolor=aspect_colors[aspect_label], facecolor='none')
        ax.add_patch(rect)
    highres_path = os.path.join(output_dir, "annotated_image.png")
    fig.savefig(highres_path, bbox_inches='tight', dpi=300)
    # --- Layout with columns ---
    col1, col2 = st.columns([3,2])
    with col1:
        st.subheader("Annotated Image")
        st.markdown("**Legend:**")
        for cls, color in aspect_colors.items():
            st.markdown(f"<span style='color:{color}; font-weight:bold;'>&#9608;</span> {cls}", unsafe_allow_html=True)
        st.image(highres_path, use_column_width=True   )
    with col2:
        st.subheader("Counts & Percentages")
        total_grains = sum(aspect_counts.values())
        counts_data = {cls: {"Count": aspect_counts.get(cls, 0), "Percent": f"{(aspect_counts.get(cls,0)/max(total_grains,1)*100):.2f}%"}
                       for cls in aspect_colors.keys()}
        df_counts = pd.DataFrame(counts_data).T
        st.table(df_counts)
        # --- Bar chart with non-zero classes and short labels ---
        non_zero_classes = [cls for cls, count in aspect_counts.items() if count > 0]
        counts_for_plot = [aspect_counts[cls] for cls in non_zero_classes]
        colors_for_plot = [aspect_colors[cls] for cls in non_zero_classes]
        x_labels = [short_labels[cls] for cls in non_zero_classes]
        fig2, ax2 = plt.subplots(figsize=(5,4))
        bars = ax2.bar(x_labels, counts_for_plot, color=colors_for_plot)
        ax2.set_ylabel("Count")
        ax2.set_title("Grain Count per Class")
        # Add dynamic labels on bars
        for bar in bars:
            height = bar.get_height()
            x_pos = bar.get_x() + bar.get_width()/2
            if height >= 200:
                y_pos = height - 2
                va = 'top'
                color = 'white'
            else:
                y_pos = height + 3
                va = 'bottom'
                color = 'black'
            ax2.text(x_pos, y_pos, f'{int(height)}', ha='center', va=va, color=color, fontweight='bold')
        # Total grains
        ax2.text(0.95, 0.95, f"Total: {total_grains}", transform=ax2.transAxes,
                 ha='right', va='top', fontsize=12, fontweight='bold', color='black')
        st.pyplot(fig2)
    # --- Annotated Image with Horizontal Legend ---
    img_pil = Image.open(highres_path)
    img_width, img_height = img_pil.size
    # Legend settings
    legend_items = list(aspect_colors.items())
    padding = 20
    box_size = 40
    spacing = 20
    font_size = 40
    # Scalable font
    try:
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()
    # Create new image with extra height for legend
    total_height = img_height + box_size + 3*padding
    new_img = Image.new("RGB", (img_width, total_height), "white")
    draw = ImageDraw.Draw(new_img)
    # Multi-row horizontal legend
    max_width = img_width - 2*padding
    x_start, y_start = padding, padding
    row_height = box_size + 10
    for label, color in legend_items:
        label_width = draw.textlength(label, font=font)
        item_width = box_size + 5 + label_width + spacing
        if x_start + item_width > max_width:
            x_start = padding
            y_start += row_height
        draw.rectangle([x_start, y_start, x_start+box_size, y_start+box_size], fill=color)
        draw.text((x_start+box_size+5, y_start), label, fill="black", font=font)
        x_start += item_width
    # Paste original image below legend
    new_img.paste(img_pil, (0, y_start + row_height))
    buf = io.BytesIO()
    new_img.save(buf, format="PNG")
    buf.seek(0)
    st.download_button(
        label="Download Annotated Image with Horizontal Legend",
        data=buf,
        file_name="rice_analysis_horizontal_legend.png",
        mime="image/png"
    )
