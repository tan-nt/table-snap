
from utils.unitable_util import plot_rec_box, LoadImage, format_html, box_4_2_poly_to_box_4_1
import time
from rapid_table.main import ModelType
from rapidocr_onnxruntime import RapidOCR
from rapid_table import RapidTable, RapidTableInput
from wired_table_rec import WiredTableRecognition
from wired_table_rec.main import WiredTableRecognition
from lineless_table_rec import LinelessTableRecognition
import cv2
from PIL import Image


img_loader = LoadImage()

unitable_table_Engine = RapidTable(RapidTableInput(model_type=ModelType.UNITABLE.value, model_path={
            "encoder": f"models/tsr/unitable_encoder.pth",
            "decoder": f"models/tsr/unitable_decoder.pth",
            "vocab": f"models/tsr/unitable_vocab.json",
        }))

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
    return unitable_table_Engine, table_engine_type


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

    return complete_html, table_boxes_img, ocr_boxes_img, polygons, all_elapse


def handle_digital_table():
    file_name = "digital_table_3"
    image_path = f"vn_tsr_dataset/digital/{file_name}/img/{file_name}.png"
    html_path = f"vn_tsr_dataset/digital/{file_name}/annotation/content.html"
    structure_path = f"vn_tsr_dataset/digital/{file_name}/annotation/structure.json"
    img = Image.open(image_path)
    img_input = img
    small_box_cut_enhance = True
    table_engine_type = "unitable"  # Example, replace with actual method
    char_ocr = True
    rotated_fix = True
    col_threshold = 15
    row_threshold = 10
    complete_html, table_boxes_img, ocr_boxes_img, polygons, all_elapse = process_image(
        img_input,
        small_box_cut_enhance,
        table_engine_type,
        char_ocr,
        rotated_fix,
        col_threshold,
        row_threshold,
    )

    # open file and write complete_html
    with open(html_path, "w") as f:
        f.write(complete_html)

    # open file and write table_boxes
    bbox_cells = '{"cells": ['
    for polygon in polygons:
        x0, y0, x1, y1 = polygon[0], polygon[1], polygon[2], polygon[3]
        x0 = round(x0)
        y0 = round(y0)
        x1 = round(x1)
        y1 = round(y1)
        bbox_cells += f'{{"x0": {x0}, "y0": {y0}, "x1": {x1}, "y1": {y1}}},'
    bbox_cells += ']}'
    with open(structure_path, "a") as f:
        f.write(bbox_cells)


def handle_printed_table():
    file_name = "printed_table_2"
    image_path = f"vn_tsr_dataset/printed/{file_name}/img/{file_name}.png"
    html_path = f"vn_tsr_dataset/printed/{file_name}/annotation/content.html"
    structure_path = f"vn_tsr_dataset/printed/{file_name}/annotation/structure.json"
    img = Image.open(image_path)
    img_input = img
    small_box_cut_enhance = True
    table_engine_type = "unitable"  # Example, replace with actual method
    char_ocr = True
    rotated_fix = True
    col_threshold = 15
    row_threshold = 10
    complete_html, table_boxes_img, ocr_boxes_img, polygons, all_elapse = process_image(
        img_input,
        small_box_cut_enhance,
        table_engine_type,
        char_ocr,
        rotated_fix,
        col_threshold,
        row_threshold,
    )

    # open file and write complete_html
    with open(html_path, "w") as f:
        f.write(complete_html)

    # open file and write table_boxes
    bbox_cells = '{"cells": ['
    for polygon in polygons:
        x0, y0, x1, y1 = polygon[0], polygon[1], polygon[2], polygon[3]
        x0 = round(x0)
        y0 = round(y0)
        x1 = round(x1)
        y1 = round(y1)
        bbox_cells += f'{{"x0": {x0}, "y0": {y0}, "x1": {x1}, "y1": {y1}}},'
    bbox_cells += ']}'
    with open(structure_path, "a") as f:
        f.write(bbox_cells)


if __name__ == "__main__":
    # handle_digital_table()
    handle_printed_table()