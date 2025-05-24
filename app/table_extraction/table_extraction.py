import cv2
import pdfplumber
import pandas as pd
import streamlit as st
import numpy as np
from easyocr import Reader

# Initialize EasyOCR
ocr_engine = Reader(['vi', 'en'], gpu=False)  # Add 'en' or other languages as needed

# Updated extract_table function
def extract_table(file, file_type):
    if file_type in ["png", "jpg", "jpeg"]:
        return extract_table_from_image(file)
    elif file_type == "pdf":
        return extract_table_from_pdf(file)
    elif file_type == "xlsx":
        return extract_table_from_excel(file)
    else:
        st.warning(f"Unsupported file type '{file_type}' for this demo.")
        return pd.DataFrame()


def extract_table_from_pdf(pdf_file):
    tables = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            if page.extract_table():
                tables.append(page.extract_table())
    # Convert tables to DataFrame
    if tables:
        df = pd.DataFrame(tables[0][1:], columns=tables[0][0])  # Use the first table
        return df
    else:
        return pd.DataFrame()  # Return empty DataFrame if no tables found


def extract_table_from_excel(excel_file):
    # Read the first sheet
    df = pd.read_excel(excel_file)
    return df


def detect_and_show_table(image_file):
    if isinstance(image_file, str):
        # If image_file is a path, read with cv2.imread
        image = cv2.imread(image_file)
    else:
        # If it's a file-like object (e.g., Streamlit file), read as bytes
        image_file.seek(0)
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Could not read image from {image_file}")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Rest of your detection code...
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    binary = 255 - binary

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
    table_mask = cv2.add(horizontal_lines, vertical_lines)

    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    table_detected_image = image.copy()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(table_detected_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    table_detected_image_rgb = cv2.cvtColor(table_detected_image, cv2.COLOR_BGR2RGB)
    return table_detected_image_rgb, contours


def extract_table_from_image(image_file, contour):
    # Handle image_file being a path or a file-like object
    if isinstance(image_file, str):
        # If image_file is a path, read it
        image = cv2.imread(image_file)
    else:
        # If it's a file-like object (e.g. from Streamlit), read as bytes
        image_file.seek(0)
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Could not read image from {image_file}")

    # Crop to the detected table region
    x, y, w, h = cv2.boundingRect(contour)
    cropped_image = image[y:y+h, x:x+w]

    # Convert to RGB for EasyOCR
    cropped_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

    # Perform OCR using EasyOCR
    results = ocr_engine.readtext(cropped_image_rgb)

    # Parse OCR results into structured data
    table_data = []
    for result in results:
        text = result[1].strip()
        if text:  # Skip empty text
            table_data.append(text)

    if not table_data:
        print("No text found in OCR.")
        return pd.DataFrame()  # Return empty DataFrame if no data

    # Optional: Split each row by whitespace or a delimiter
    rows = [row.split() for row in table_data]

    # Convert to DataFrame
    df = pd.DataFrame(rows)
    return df