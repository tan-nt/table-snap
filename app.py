import streamlit as st
from streamlit_option_menu import option_menu
from app.table_extraction.table_extraction import detect_and_show_table, extract_table_from_image
import os
from io import BytesIO
import cv2
from utils.image import image_to_base64
import psutil


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
        options=["🏠 Home", "📂 Table Structure Extraction", "💬 Table Chatbot"],
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
elif selected == "💬 Table Chatbot":
    st.session_state.page = "table_chatbot"


from PIL import Image
from transformers import DetrImageProcessor
from transformers import TableTransformerForObjectDetection

import torch
import matplotlib.pyplot as plt
import os
import time
from transformers import DetrFeatureExtractor
import pandas as pd
import pytesseract
from utils.unitable_util import plot_rec_box, LoadImage, format_html, box_4_2_poly_to_box_4_1
import time
from lineless_table_rec import LinelessTableRecognition
from rapid_table import RapidTable, RapidTableInput
from wired_table_rec import WiredTableRecognition
from rapid_table.main import ModelType
# from table_cls import TableCls
from rapidocr_onnxruntime import RapidOCR
from wired_table_rec.main import WiredTableInput, WiredTableRecognition
from lineless_table_rec.main import LinelessTableInput, LinelessTableRecognition

cell_recognition_model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition")
table_detection_model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")

feature_extractor = DetrFeatureExtractor()

img_loader = LoadImage()
# table_cls_ins = TableCls()

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


COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]


def plot_results(model, pil_img, scores, labels, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for score, label, (xmin, ymin, xmax, ymax),c  in zip(scores.tolist(), labels.tolist(), boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        text = f'{model.config.id2label[label]}: {score:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    st.pyplot(plt)
    # plt.show()


def table_detection(image):
    image = Image.open(image).convert("RGB")
    width, height = image.size
    image.resize((int(width*0.5), int(height*0.5)))

    feature_extractor = DetrImageProcessor()
    encoding = feature_extractor(image, return_tensors="pt")

    with torch.no_grad():
        outputs = table_detection_model(**encoding)

    width, height = image.size
    results = feature_extractor.post_process_object_detection(outputs, threshold=0.7, target_sizes=[(height, width)])[0]
    plot_results(table_detection_model, image, results['scores'], results['labels'], results['boxes'])
    return results['boxes']


def cell_detection(file_path):
    image = Image.open(file_path).convert("RGB")
    width, height = image.size
    image.resize((int(width*0.5), int(height*0.5)))


    encoding = feature_extractor(image, return_tensors="pt")
    encoding.keys()

    with torch.no_grad():
      outputs = cell_recognition_model(**encoding)


    target_sizes = [image.size[::-1]]
    results = feature_extractor.post_process_object_detection(outputs, threshold=0.6, target_sizes=target_sizes)[0]
    plot_results(cell_recognition_model, image, results['scores'], results['labels'], results['boxes'])
    cell_recognition_model.config.id2label

def plot_results_specific(pil_img, scores, labels, boxes,lab):
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for score, label, (xmin, ymin, xmax, ymax), c in zip(scores.tolist(), labels.tolist(), boxes.tolist(), colors):
        if label == lab:
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                       fill=False, color=c, linewidth=3))
            text = f'{table_detection_model.config.id2label[label]}: {score:0.2f}'
            ax.text(xmin, ymin, text, fontsize=15,
                    bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    st.pyplot(plt)

def draw_box_specific(image_path,labelnum):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    encoding = feature_extractor(image, return_tensors="pt")

    with torch.no_grad():
        outputs = cell_recognition_model(**encoding)

    results = feature_extractor.post_process_object_detection(outputs, threshold=0.7, target_sizes=[(height, width)])[0]
    plot_results_specific(image, results['scores'], results['labels'], results['boxes'],labelnum)

def compute_boxes(image_path):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    encoding = feature_extractor(image, return_tensors="pt")

    with torch.no_grad():
        outputs = cell_recognition_model(**encoding)

    results = feature_extractor.post_process_object_detection(outputs, threshold=0.7, target_sizes=[(height, width)])[0]
    # print('results',results)
    boxes = results['boxes'].tolist()
    labels = results['labels'].tolist()

    return boxes,labels

def extract_table(image_path):
    image = Image.open(image_path).convert("RGB")
    boxes,labels = compute_boxes(image_path)

    cell_locations = []
    # print('boxes',boxes)
    # print('labels',labels)

    for box_row, label_row in zip(boxes, labels):
        if label_row == 2:
            for box_col, label_col in zip(boxes, labels):
                print('box_col',box_col)
                print('label_col',label_col)
                if label_col == 1:
                    cell_box = (box_col[0], box_row[1], box_col[2], box_row[3])
                    cell_locations.append(cell_box)

    cell_locations.sort(key=lambda x: (x[1], x[0]))

    num_columns = 0
    box_old = cell_locations[0]

    for box in cell_locations[1:]:
        x1, y1, x2, y2 = box
        x1_old, y1_old, x2_old, y2_old = box_old
        num_columns += 1
        if y1 > y1_old:
            break

        box_old = box

    headers = []
    for box in cell_locations[:num_columns]:
        x1, y1, x2, y2 = box
        cell_image = image.crop((x1, y1, x2, y2))
        new_width = cell_image.width * 4
        new_height = cell_image.height * 4
        cell_image = cell_image.resize((new_width, new_height), resample=Image.LANCZOS)
        cell_text = pytesseract.image_to_string(cell_image)
        headers.append(cell_text.rstrip())

    df = pd.DataFrame(columns=headers)

    row = []
    for box in cell_locations[num_columns:]:
        x1, y1, x2, y2 = box
        cell_image = image.crop((x1, y1, x2, y2))
        new_width = cell_image.width * 4
        new_height = cell_image.height * 4
        cell_image = cell_image.resize((new_width, new_height), resample=Image.LANCZOS)
        cell_text = pytesseract.image_to_string(cell_image)

        if len(cell_text) > num_columns:
            cell_text = cell_text[:num_columns]

        row.append(cell_text.rstrip())

        if len(row) == num_columns:
            df.loc[len(df)] = row
            row = []

    return df

from bs4 import BeautifulSoup

def normalize_rows_and_headers(headers, rows):
    # If there are no rows, return None, None
    if len(rows) == 0:
        return None, None

    if len(headers) == 0:
        headers = rows[0]
        rows = rows[1:]

    # Determine the maximum number of columns in the rows
    max_columns = max(len(row) for row in rows)

    # Ensure the headers match the maximum number of columns
    if len(headers) < max_columns:
        headers.extend([''] * (max_columns - len(headers)))

    # Normalize rows to match header length
    for row in rows:
        if len(row) != len(headers):
            row.extend([''] * (len(headers) - len(row)))  # Add empty strings to rows with fewer columns

    return headers, rows


def html_to_csv(html):
    print('html', html)
    # Step 1: Parse the HTML content
    soup = BeautifulSoup(html, 'html.parser')

    # Step 2: Find the table in the HTML (assuming the first <table> is what you want)
    table = soup.find('table')

    # Step 3: Extract the headers (th elements)
    headers = [header.text.strip() for header in table.find_all('th')]

    # Step 4: Extract the rows (tr elements)
    rows = []
    for row in table.find_all('tr')[1:]:  # Skip the header row
        columns = row.find_all('td')
        if columns:
            rows.append([col.text.strip() for col in columns])

    headers, rows = normalize_rows_and_headers(headers, rows)

    print('headers', headers)
    print('rows', rows)
    # Step 5: Create a DataFrame from the extracted data
    df = pd.DataFrame(rows, columns=headers)

    # Step 6: Save DataFrame to CSV
    csv_filename = "table_output.csv"
    df.to_csv(csv_filename, index=False)
    return df


def upload_and_extract_table():
    global latest_df
    # Sidebar Configuration UI
    st.sidebar.title("Configuration")
    structure_extraction_method = st.sidebar.radio(
        "Choose extraction method",
        options=["Contours Detection", "UniTable", "Table Transformer"],
        index=1,
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
            cls, elasp = table_cls_ins(img)
            if cls == 'wired':
                table_type = 'wired_table_v2'
            all_elapse = f"- Table Type: {table_type}\n - Table all cost: {sum_elapse:.5f} seconds"

            if not table.empty:
                print(f'set latest_df={table}')
                st.session_state.latest_df = table

            st.sidebar.write(all_elapse)
        elif structure_extraction_method == "UniTable":
            st.sidebar.title("Elapsed Time")
            from PIL import Image
            img = Image.open(file)
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
            st.markdown(complete_html, unsafe_allow_html=True)
            table = html_to_csv(complete_html)
            if not table.empty:
                st.session_state.latest_df = table

            st.title("Table Boxes")
            st.image(table_boxes_img, caption="Detected Table Boxes", use_column_width=True)

            st.title("OCR Boxes")
            st.image(ocr_boxes_img, caption="Detected OCR Boxes", use_column_width=True)

            st.sidebar.write(all_elapse)
        elif structure_extraction_method == "Table Transformer":
            st.sidebar.title("Elapsed Time")
            start_time = time.time()
            st.title("Table Detection result")
            pred_bbox = table_detection(file)

            st.title("Table Structure Recognition result")
            image_path = "assets/tsr_examples/invoice.jpg"
            cell_detection(image_path)
            # draw_box_specific(image_path,1)
            df = extract_table(image_path)
            st.write(df)
            # df = extract_table(image_path)
            # df.to_csv('data.csv', index=False)

            end_time = time.time()
            sum_elapse = end_time - start_time
            table_type = 'lineless_table'
            cls, elasp = table_cls_ins(img)
            if cls == 'wired':
                table_type = 'wired_table_v2'
            all_elapse = f"- Table Type: {table_type}\n - Table all cost: {sum_elapse:.5f} seconds"
            if not df.empty:
                print(f'set latest_df={df}')
                st.session_state.latest_df = df
            st.sidebar.write(all_elapse)
        else:
            st.sidebar.warning("No engine selected.")
        st.sidebar.info("The engine has finished the task. Please check the result below.")


def select_table_model(img, table_engine_type, det_model, rec_model):
    cls, elasp = table_cls_ins(img)
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

from transformers import AutoTokenizer, AutoModelForCausalLM
model_name = "tablegpt/TableGPT2-7B"
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype='auto', device_map='auto')
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.float32,  # Force the model to use float32 precision
#     device_map='auto'  # Automatically distribute the model across available devices
# )

from accelerate import disk_offload
# Load the model (with device_map="auto" and offloading enabled)
table_gpt_model = AutoModelForCausalLM.from_pretrained(
    "gpt2",  # Replace with your model name
    torch_dtype="auto",  # Auto-detect precision
    device_map="auto",  # Automatically assign devices (GPU/CPU)
    offload_folder="./TableGPT2-7B"  # Specify the folder for offloading model weights
)

# Specify the offload directory (the folder where the model weights will be saved)
offload_directory = './TableGPT2-7B'  # You can change this path to your desired folder

# Call the disk_offload function to offload the model weights to disk
disk_offload(table_gpt_model, offload_dir=offload_directory)

tokenizer = AutoTokenizer.from_pretrained(model_name)

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

elif st.session_state.page == "table_chatbot":
    st.title("💬 Chat about your tables with AI")
    st.markdown("Ask me anything about your extracted data or tables!")
    # Initialize session state to store conversation history
    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    st.info("We got the latest Table:")
    latest_df = st.session_state.get("latest_df", None)  # Default to None if not set
    if latest_df is not None:
        # Process the data
        st.write("Processing Data:")
        st.write(latest_df)
    else:
        st.write("No data to process.")


    # Text input for the user
    user_input = st.text_input("Type your question:")

    if user_input:
        # Append user input to the conversation history
        st.session_state.conversation.append(f"User: {user_input}")

        prompt_template = """
            Given access to several pandas dataframes, answer the user's question.
            /*
                "{var_name}.head(5).to_string(index_False)" as follows:
                {df_info}
            */
            Question: {user_input}
        """

        if latest_df is not None:
            prompt = prompt_template.format(var_name="latest_df", df_info=latest_df.head(5).to_string(index=False), user_input=user_input)
        else:
            prompt = user_input
        messages = [
            {"role": "system", "content": "You are a helpful assistant that can answer questions about the table."},
            {"role": "user", "content": prompt}
        ]

       # Set the device (either CUDA for GPU or CPU)
        device = "cuda" if torch.cuda.is_available() else "cpu"


        # Set the device (either CUDA for GPU or CPU)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize the model using .to_empty() and specify the device
        table_gpt_model.to_empty(device=device)  # Specify the device here (cuda or cpu)

        # Now move the model to the correct device (either CPU or GPU)
        table_gpt_model = table_gpt_model.to(device)

        # Tokenize input and prepare model inputs
        model_inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)

        # Generate response from the model
        generated_ids = table_gpt_model.generate(**model_inputs, max_new_tokens=512)

        # Decode the response
        answer = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print('answer=', answer)

        # Append bot response to the conversation history
        bot_response = f"🤖 Bot: {answer}"
        st.session_state.conversation.append(bot_response)

        # Display the entire conversation history
        for message in st.session_state.conversation:
            st.write(message)


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