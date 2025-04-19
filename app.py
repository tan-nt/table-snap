import streamlit as st
from streamlit_option_menu import option_menu
from app.table_extraction.table_extraction import detect_and_show_table, extract_table_from_image
import os
from io import BytesIO
import cv2
from utils.image import image_to_base64


st.set_page_config(page_title="Table Snap", layout="wide", page_icon="assets/table_snap_icon.png")
st.title("Table Snap - A solution for extracting tables from invoices, receipts, and more.")
st.markdown(
    f"""
    <div style="display: flex; align-items: center; justify-content: space-between;">
        <div style="flex: 1;">
            <p style="font-size: 16px; margin: 0;">
                Manual extraction of table data from invoices and receipts is tedious, error-prone, and not scalable.
                TableSnap automates table detection and structure recognition using deep learning, enabling accurate extraction, organization, and analysis - even in complex or Vietnamese documents
            </p>
        </div>
        <div style="flex-shrink: 0;">
            <img src="data:image/png;base64,{image_to_base64("assets/table_snap_icon.png")}" width="100"/>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


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
    st.title("Welcome to Table Snap")
    st.markdown(
        """
        ### ✍️ Let me share a story

        A few years ago, I took a part-time job where I spent hours manually extracting table data from invoices and receipts — copying rows from images into CSVs.
        🌀 It was repetitive. 😤 Frustrating. ❌ Easy to mess up.

        Then it hit me: **I’m not the only one.**
        👨‍💼 Accountants, 🧑‍🔬 researchers, 👩‍💻 office workers — so many people waste **precious time** on this boring, error-prone task.
        ⏳ Time that could be used for **thinking**, **building**, or **solving bigger problems**.

        ---

        ### 🚀 That’s why we built **TableSnap**

        🔍 A full-stack, deep learning–powered app that **automatically detects and extracts** structured tables from documents with **high accuracy**.
        🧠 Built with cutting-edge models to turn invoices, receipts, and messy scans into clean, query-ready data.

        ### 🤖 But we didn’t stop there

        💬 We added an AI chatbot that:
        - Summarizes key insights 📊
        - Generates SQL-ready data schemas 🧾
        - Answers your questions in natural language 🧠

        From raw documents ➡️ to analytics ➡️ to insights — in one seamless flow.

        ### 💡 Moreover, table snap is an open source project.

        We open-sourced the project so anyone can learn, build on it, or use it — especially in research and non-profit contexts.
        🙌 Because **freeing human time** is worth it.

        ### 🧷 TableSnap isn’t just a tool —
        it’s a mission to **turn wasted hours into meaningful work**.
        """
    )


elif st.session_state.page == "upload_extract":
    upload_and_extract_table()

elif st.session_state.page == "chat":
    st.title("💬 Chat with AI")
    st.markdown("Ask me anything about your extracted data or tables!")
    user_input = st.text_input("Type your question:")
    if user_input:
        st.write(f"🤖 Bot: I'm here to help with your data questions!")

st.markdown("---")
st.markdown(
    f"""
    <div style="text-align: right;">
        <p style="margin-bottom: 5px;"><strong>Developed by IMP student</strong> – Powered by Streamlit 🚀</p>
        <img src="data:image/png;base64,{image_to_base64("assets/imp.png")}" width="100"/>
    </div>
    """,
    unsafe_allow_html=True
)