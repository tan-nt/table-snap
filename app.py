import streamlit as st
from streamlit_option_menu import option_menu
from app.table_extraction.table_extraction import detect_and_show_table, extract_table_from_image
import os
from io import BytesIO
import cv2

st.set_page_config(page_title="TableSnap", layout="wide", page_icon="🧾")
st.title("🧾 TableSnap")
st.markdown("**Effortlessly extract, organize, and analyze tables from invoices, receipts, and more.**")

st.sidebar.header("Navigation")
with st.sidebar:
    selected = option_menu(
        menu_title=None,
        options=["🏠 Home", "📂 Upload & Extract", "💬 Go Chat"],
        icons=["house", "file-earmark-arrow-up", "chat-dots"],
        menu_icon="cast",
        default_index=1,
        styles={
            "container": {"padding": "0!important", "background-color": "#f0f2f6"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "5px",
                "--hover-color": "#eee",
            },
            "nav-link-selected": {"background-color": "#02ab21"},
        },
    )

if "page" not in st.session_state:
    st.session_state.page = "upload_extract"  
# Default to "Upload & Extract" page

if selected == "🏠 Home":
    st.session_state.page = "home"
elif selected == "📂 Upload & Extract":
    st.session_state.page = "upload_extract"
elif selected == "💬 Go Chat":
    st.session_state.page = "chat"

def upload_and_extract_table():
    # Sidebar Configuration UI
    st.sidebar.title("Table Extraction Configuration")
    structure_extraction_method = st.sidebar.radio(
        "Choose Table Extraction Method",
        options=["Contours Detection", "Yolo", "Transformer"],
        help="Select how you want to extract tables from the uploaded image."
    )
    structure_extraction_method = st.sidebar.radio(
        "Choose Table Structure Extraction Method",
        options=["Contours Detection", "Yolo", "Transformer"],
        help="Select how you want to extract tables from the uploaded image."
    )
    
    st.header("📂 Upload and Extract Table")
    
    file = st.file_uploader("Upload a PDF, Image, or Excel File", type=["pdf", "png", "jpg", "jpeg", "xlsx"])
    file_type = None 
    if not file:
        st.info("No file uploaded. Using the default image for table extraction.")
        default_file_path = os.path.join("assets", "invoice/vi_invoice.jpg")
        with open(default_file_path, "rb") as default_file:
            file_content = default_file.read()  
        file = BytesIO(file_content)  
        file_type = "jpg"  
    else:
        file_type = file.name.split(".")[-1].lower()

    if file_type in ["png", "jpg", "jpeg"]:
        st.image(file, caption="Uploaded Image", use_container_width=True)
    else:
        st.warning("Displaying images is supported only for PNG, JPG, and JPEG formats.")
        
    if structure_extraction_method == "Contours Detection":
        table_detected_image, contours = detect_and_show_table(file)
        st.image(table_detected_image, caption="Detected Table(s)", use_container_width=True)

        if contours:
            st.info(f"{len(contours)} table(s) detected. Processing the largest table.")            
            largest_contour = max(contours, key=cv2.contourArea) # Process the largest table
            table = extract_table_from_image(file, largest_contour)   
            if not table.empty:
                st.write("Extracted Table:")
                st.write(table)                
                st.download_button("Download as CSV", table.to_csv(index=False), "table.csv")
            else:
                st.warning("No text detected in the table region.")
    else:
        st.warning("No tables detected in the uploaded image.")

if st.session_state.page == "home":
    st.title("Welcome to TableSnap! 🧾")
    st.markdown(
        """
        **TableSnap** is your all-in-one solution for extracting, summarizing, 
        and analyzing data from tables in invoices, receipts, and other documents.
        """
    )
    st.image("https://source.unsplash.com/featured/?data", use_container_width=True)

elif st.session_state.page == "upload_extract":
    upload_and_extract_table()

elif st.session_state.page == "chat":
    st.title("💬 Chat with AI")
    st.markdown("Ask me anything about your extracted data or tables!")
    user_input = st.text_input("Type your question:")
    if user_input:
        st.write(f"🤖 Bot: I'm here to help with your data questions!")

st.markdown("---")
st.markdown("**Developed by TableSnap** - Powered by Streamlit 🚀")
