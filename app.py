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
        options=["🏠 Home", "📂 Table Structure Extraction", "💬 Chatbot"],
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
    st.session_state.page = "table_structure_extraction"
# Default to "Upload & Extract" page

if selected == "🏠 Home":
    st.session_state.page = "home"
elif selected == "📂 Table Structure Extraction":
    st.session_state.page = "table_structure_extraction"
elif selected == "💬 Chatbot":
    st.session_state.page = "chat"

def upload_and_extract_table():
    # Sidebar Configuration UI
    st.sidebar.title("Configuration")
    structure_extraction_method = st.sidebar.radio(
        "Choose extraction method",
        options=["Contours Detection", "UniTable", "Table Transformer"],
        help="Select how you want to extract tables from the uploaded image.",
    )

    st.header("📂 Upload and Extract Table")
    st.info("Choose an image and click 'Run' on the sidebar to extract the table.")

    example_images = [
        {"label": "Invoice Data", "path": "assets/tsr_examples/invoice.jpg"},
        {"label": "Example 2", "path": "assets/tsr_examples/en_invoice.png"},
        {"label": "Example 3", "path": "assets/tsr_examples/handwritten_invoice.jpeg"},
        {"label": "Example 4", "path": "assets/tsr_examples/tsr_table.jpg"},
    ]

    from streamlit_image_select import image_select
    img = image_select("Table image examples", [image['path'] for image in example_images])
    st.write(img)

    # Allow the user to upload their own image
    file = st.file_uploader("Upload a PDF, Image, or Excel File", type=["pdf", "png", "jpg", "jpeg", "xlsx"])

    # Process the uploaded file or use the selected example image
    file_type = None
    if not file:
        with open(img, "rb") as default_file:
            file_content = default_file.read()
        file = BytesIO(file_content)
        file_type = img.split(".")[-1].lower()
    else:
        file_type = file.name.split(".")[-1].lower()

    if file_type in ["png", "jpg", "jpeg"]:
        st.image(file, caption="Uploaded Image", use_container_width=True)
    else:
        st.warning("Displaying images is supported only for PNG, JPG, and JPEG formats.")

    if st.sidebar.button('Run'):
        st.sidebar.info("The engine is running to serve you the best result...")

        st.info("The result will be displayed here")
        if structure_extraction_method == "Contours Detection":
            st.sidebar.title("Elapsed Time")
            start_time = time.time()
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
            sum_elapse = time.time() - start_time
            table_type = 'lineless_table'
            cls, elasp = table_cls(img)
            if cls == 'wired':
                table_type = 'wired_table_v2'
            all_elapse = f"- Table Type: {table_type}\n - Table all cost: {sum_elapse:.5f} seconds"
            st.sidebar.write(all_elapse)
        elif structure_extraction_method == "UniTable":
            st.sidebar.title("Elapsed Time")
            from PIL import Image
            img = Image.open(file)
            # st.info(f'Data={process_image(img, False, "unitable", True, True, 15, 10)}')
            img_input = img
            small_box_cut_enhance = True
            table_engine_type = "unitable"  # Example, replace with actual method
            char_ocr = True
            rotated_fix = True
            col_threshold = 15
            row_threshold = 10

            complete_html, table_boxes_img, ocr_boxes_img, all_elapse = process_image(
                img_input,
                small_box_cut_enhance,
                table_engine_type,
                char_ocr,
                rotated_fix,
                col_threshold,
                row_threshold,
            )

            st.title("Html Render")
            st.markdown(complete_html, unsafe_allow_html=True)  # Rendering the HTML content

            st.title("Table Boxes")
            st.image(table_boxes_img, caption="Detected Table Boxes", use_column_width=True)

            st.title("OCR Boxes")
            st.image(ocr_boxes_img, caption="Detected OCR Boxes", use_column_width=True)

            st.sidebar.write(all_elapse)
        elif structure_extraction_method == "Table Transformer":
            st.sidebar.warning("Table Transformer is not implemented yet.")
        else:
            st.sidebar.warning("No engine selected.")
        st.sidebar.info("The engine has finished the task. Please check the result below.")


from utils.unitable_util import plot_rec_box, LoadImage, format_html, box_4_2_poly_to_box_4_1
import time
from lineless_table_rec import LinelessTableRecognition
from rapid_table import RapidTable, RapidTableInput
from wired_table_rec import WiredTableRecognition
from rapid_table.main import ModelType
from table_cls import TableCls
from rapidocr_onnxruntime import RapidOCR


img_loader = LoadImage()
table_cls = TableCls()

from wired_table_rec.main import WiredTableInput, WiredTableRecognition
from lineless_table_rec.main import LinelessTableInput, LinelessTableRecognition

rapid_table_engine = RapidTable(RapidTableInput(model_type=ModelType.PPSTRUCTURE_ZH.value, model_path="models/tsr/ch_ppstructure_mobile_v2_SLANet.onnx"))
SLANet_plus_table_Engine = RapidTable(RapidTableInput(model_type=ModelType.SLANETPLUS.value, model_path="models/tsr/slanet-plus.onnx"))
unitable_table_Engine = RapidTable(RapidTableInput(model_type=ModelType.UNITABLE.value, model_path={
            "encoder": f"models/tsr/unitable_encoder.pth",
            "decoder": f"models/tsr/unitable_decoder.pth",
            "vocab": f"models/tsr/unitable_vocab.json",
        }))

wired_input = WiredTableInput()
lineless_input = LinelessTableInput()
wired_engine = WiredTableRecognition(wired_input)
wired_table_engine_v1 = wired_engine
wired_table_engine_v2 = wired_engine
lineless_engine = LinelessTableRecognition(lineless_input)
det_model_dir = {
    "mobile_det": "models/ocr/ch_PP-OCRv4_det_infer.onnx",
}

rec_model_dir = {
    "mobile_rec": "models/ocr/ch_PP-OCRv4_rec_infer.onnx",
}
ocr_engine_dict = {}
for det_model in det_model_dir.keys():
    for rec_model in rec_model_dir.keys():
        det_model_path = det_model_dir[det_model]
        rec_model_path = rec_model_dir[rec_model]
        key = f"{det_model}_{rec_model}"
        ocr_engine_dict[key] = RapidOCR(det_model_path=det_model_path, rec_model_path=rec_model_path)

def select_table_model(img, table_engine_type, det_model, rec_model):
    cls, elasp = table_cls(img)
    if cls == 'wired':
        return unitable_table_Engine, "wired_table_v2"
    elif cls == 'lineless':
        return unitable_table_Engine, "lineless_table"
    else:
        return unitable_table_Engine, table_engine_type
    # if table_engine_type == "RapidTable(SLANet)":
    #     return rapid_table_engine, table_engine_type
    # elif table_engine_type == "RapidTable(SLANet-plus)":
    #     return SLANet_plus_table_Engine, table_engine_type
    # elif table_engine_type == "RapidTable(unitable)":
    #     return unitable_table_Engine, table_engine_type
    # elif table_engine_type == "wired_table_v1":
    #     return wired_table_engine_v1, table_engine_type
    # elif table_engine_type == "wired_table_v2":
    #     print("使用v2 wired table")
    #     return wired_table_engine_v2, table_engine_type
    # elif table_engine_type == "lineless_table":
    #     return lineless_table_engine, table_engine_type
    # elif table_engine_type == "auto":
    #     cls, elasp = table_cls(img)
    #     if cls == 'wired':
    #         table_engine = wired_table_engine_v2
    #         return table_engine, "wired_table_v2"
    #     return lineless_table_engine, "lineless_table"

def select_ocr_model(det_model, rec_model):
    return ocr_engine_dict[f"{det_model}_{rec_model}"]

def trans_char_ocr_res(ocr_res):
    word_result = []
    for res in ocr_res:
        score = res[2]
        for word_box, word in zip(res[3], res[4]):
            word_res = []
            word_res.append(word_box)
            word_res.append(word)
            word_res.append(score)
            word_result.append(word_res)
    return word_result

def process_image(img_input, small_box_cut_enhance, table_engine_type, char_ocr, rotated_fix, col_threshold, row_threshold):
    det_model="mobile_det"
    rec_model="mobile_rec"
    img = img_loader(img_input)
    start = time.time()
    table_engine, table_type = select_table_model(img, table_engine_type, det_model, rec_model)
    ocr_engine = select_ocr_model(det_model, rec_model)

    ocr_res, ocr_infer_elapse = ocr_engine(img, return_word_box=char_ocr)
    det_cost, cls_cost, rec_cost = ocr_infer_elapse
    if char_ocr:
        ocr_res = trans_char_ocr_res(ocr_res)
    ocr_boxes = [box_4_2_poly_to_box_4_1(ori_ocr[0]) for ori_ocr in ocr_res]
    if isinstance(table_engine, RapidTable):
        table_results = table_engine(img, ocr_res)
        html, polygons, table_rec_elapse = table_results.pred_html, table_results.cell_bboxes,table_results.elapse
        polygons = [[polygon[0], polygon[1], polygon[4], polygon[5]] for polygon in polygons]
    elif isinstance(table_engine, (WiredTableRecognition, LinelessTableRecognition)):
        html, table_rec_elapse, polygons, logic_points, ocr_res = table_engine(img, ocr_result=ocr_res,
                                                                                   enhance_box_line=small_box_cut_enhance,
                                                                                   rotated_fix=rotated_fix,
                                                                                   col_threshold=col_threshold,
                                                                                   row_threshold=row_threshold)
    sum_elapse = time.time() - start
    all_elapse = f"- Table Type: {table_type}\n - Table all cost: {sum_elapse:.5f} seconds\n - Table rec cost: {table_rec_elapse:.5f} seconds\n - OCR cost: {det_cost + cls_cost + rec_cost:.5f} seconds"

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    table_boxes_img = plot_rec_box(img.copy(), polygons)
    ocr_boxes_img = plot_rec_box(img.copy(), ocr_boxes)
    complete_html = format_html(html)

    return complete_html, table_boxes_img, ocr_boxes_img, all_elapse

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

elif st.session_state.page == "chat":
    st.title("💬 Chat with AI")
    st.markdown("Ask me anything about your extracted data or tables!")
    user_input = st.text_input("Type your question:")
    if user_input:
        st.write(f"🤖 Bot: I'm here to help with your data questions!")

if st.session_state.page == "table_structure_extraction":
    upload_and_extract_table()

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