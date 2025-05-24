import os
import random
import cv2
import shutil

import albumentations as A
from albumentations.core.composition import OneOf
from albumentations.core.composition import Compose
from albumentations.core.transforms_interface import ImageOnlyTransform
import random

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

import numpy as np
from utils.chat import get_google_gemini_generate_answer_v2


def build_dataset():
    digital_folder_path = 'vn_tsr_dataset/original_dataset/digital'
    print(f'digital_folder_path: {digital_folder_path}')
    printed_folder_path = 'vn_tsr_dataset/original_dataset/printed'
    print(f'printed_folder_path: {printed_folder_path}')

    start_index_number = 3

    os.makedirs(printed_folder_path, exist_ok=True)

    printed_file_list = os.listdir(printed_folder_path)
    current_index = start_index_number

    for printed_file in printed_file_list:
        printed_file_path = os.path.join(printed_folder_path, printed_file)

        if os.path.isfile(printed_file_path):
            img = cv2.imread(printed_file_path)
            if img is None:
                print(f"Warning: Could not read {printed_file_path}. Skipping.")
                continue

            # Create output file name
            six_digit_str = str(current_index).zfill(6)
            output_file_name = f"printed_table_{six_digit_str}.png"
            output_file_path = os.path.join(printed_folder_path, output_file_name)

            cv2.imwrite(output_file_path, img)
            print(f"Converted and saved: {output_file_path}")

            current_index += 1

    print("Conversion and renaming completed.")

def save_file_annotations():
    printed_folder_path = 'vn_tsr_dataset/original_dataset/printed'
    print(f'printed_folder_path: {printed_folder_path}')

    printed_file_list = os.listdir(printed_folder_path)
    for printed_file in printed_file_list:
        if printed_file.endswith('.png') and printed_file.startswith('printed_table'):
            printed_file_path = os.path.join(printed_folder_path, printed_file)
            if not os.path.isfile(printed_file_path):
                print(f"printed_file_path: {printed_file_path} not exist")
                continue
            print(f"processing printed_file_path: {printed_file_path}")
            img = cv2.imread(printed_file_path)
            if img is None:
                print(f"Warning: Could not read {printed_file_path}. Skipping.")
                continue

            six_digit_str = printed_file.split('_')[1].split('.')[0]
            destination_folder_path = f'vn_tsr_dataset/printed/printed_table_{six_digit_str}'
            os.mkdir(destination_folder_path)
            os.mkdir(f'{destination_folder_path}/img')
            os.mkdir(f'{destination_folder_path}/annotation')

            shutil.copy(printed_file_path, f'{destination_folder_path}/img/{printed_file}')
            handle_printed_table(dataset_folder='vn_tsr_dataset/printed', file_name=f"printed_table_{six_digit_str}")


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

def handle_printed_table(dataset_folder='vn_tsr_dataset/printed', file_name='printed_table_2'):
    image_path = f"{dataset_folder}/{file_name}/img/{file_name}.png"
    html_path = f"{dataset_folder}/{file_name}/annotation/content.html"
    structure_path = f"{dataset_folder}/{file_name}/annotation/structure.json"
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
        question = f"Please correct the html table for vietnamese language and only return the table in HTML format. The table is: '{complete_html}'"
        complete_html_answer = get_google_gemini_generate_answer_v2(question)
        print(f'answer: {complete_html_answer}')
        f.write(complete_html_answer)

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

if __name__ == '__main__':
    build_dataset()
    save_file_annotations()