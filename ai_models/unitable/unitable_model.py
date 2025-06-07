import time
from utils.unitable_util import plot_rec_box, LoadImage, format_html, box_4_2_poly_to_box_4_1
from rapid_table import RapidTable, RapidTableInput
from rapidocr_onnxruntime import RapidOCR
from wired_table_rec.main import WiredTableInput, WiredTableRecognition
from lineless_table_rec.main import LinelessTableInput, LinelessTableRecognition
from rapid_table.main import ModelType
import cv2
import os
from utils.ted_evaluation import TEDS
from utils.normalization import normalize_html
import csv


img_loader = LoadImage()
rapid_table_engine = RapidTable(RapidTableInput(model_type=ModelType.PPSTRUCTURE_ZH.value, model_path="model_weights/tsr/ch_ppstructure_mobile_v2_SLANet.onnx"))
SLANet_plus_table_Engine = RapidTable(RapidTableInput(model_type=ModelType.SLANETPLUS.value, model_path="model_weights/tsr/slanet-plus.onnx"))
unitable_table_Engine = RapidTable(RapidTableInput(model_type=ModelType.UNITABLE.value, model_path={
            "encoder": f"model_weights/tsr/unitable_encoder.pth",
            "decoder": f"model_weights/tsr/unitable_decoder.pth",
            "vocab": f"model_weights/tsr/unitable_vocab.json",
        }))

wired_input = WiredTableInput()
lineless_input = LinelessTableInput()
wired_engine = WiredTableRecognition(wired_input)
wired_table_engine_v1 = wired_engine
wired_table_engine_v2 = wired_engine
lineless_engine = LinelessTableRecognition(lineless_input)
det_model_dir = {
    "mobile_det": "model_weights/ocr/ch_PP-OCRv4_det_infer.onnx",
}

rec_model_dir = {
    "mobile_rec": "model_weights/ocr/ch_PP-OCRv4_rec_infer.onnx",
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

    return complete_html, table_boxes_img, ocr_boxes_img, all_elapse


def test_vn_tsr_dataset_by_unitable():
    dataset_folders = 'vn_tsr_dataset'
    teds = TEDS(structure_only=True)
    csv_filename = "ai_models/unitable/unitable_teds_results.csv"
    fieldnames = ['image_file', 'ted_score', 'pred_html', 'annotation_html']

    existing_results = {}
    if os.path.exists(csv_filename):
        with open(csv_filename, mode='r', encoding='utf-8') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                existing_results[row['image_file']] = row

    with open(csv_filename, mode='a', encoding='utf-8', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not existing_results:
            writer.writeheader()

        index = len(existing_results) + 1
        avg_time = 0
        count = 0
        for dataset_folder in os.listdir(dataset_folders):
            if not os.path.isdir(f'{dataset_folders}/{dataset_folder}'):
                continue
            if dataset_folder not in ['digital', 'printed']:
                continue

            for folder_data in os.listdir(f'{dataset_folders}/{dataset_folder}'):
                img_path = f'{dataset_folders}/{dataset_folder}/{folder_data}/img/{folder_data}.png'
                if 'table' not in img_path:
                    continue
                # if img_path in existing_results:
                #     print(f"Skipping already processed file: {img_path}")
                #     continue

                img = img_loader(img_path)
                small_box_cut_enhance = True
                table_engine_type = "unitable"
                char_ocr = True
                rotated_fix = True
                col_threshold = 15
                row_threshold = 10
                start_time = time.time()
                pred_html, table_boxes_img, ocr_boxes_img, all_elapse = process_image(
                    img,
                    small_box_cut_enhance,
                    table_engine_type,
                    char_ocr,
                    rotated_fix,
                    col_threshold,
                    row_threshold,
                )
                end_time = time.time()
                avg_time += end_time - start_time
                count += 1
                step_avg_time = avg_time / count
                # Calculate FPS
                fps = 1.0 / step_avg_time
                print(f"Processed in {step_avg_time:.4f} seconds — FPS: {fps:.2f}")

                print(f"Time taken: {end_time - start_time} seconds")
                anno_html = open(f'{dataset_folders}/{dataset_folder}/{folder_data}/annotation/content.html', 'r', encoding='utf-8').read()
                normalized_pred_html = normalize_html(pred_html)
                score = teds.evaluate(normalized_pred_html, anno_html)
                print(f"TEDS Score: {score}, Index: {index}, Image Path: {img_path}")
                index += 1

                # Save to CSV immediately
                writer.writerow({
                    "image_file": img_path,
                    "ted_score": score,
                    "pred_html": normalized_pred_html,
                    "annotation_html": anno_html
                })
                csv_file.flush()  # Force write to disk after each entry



    print(f"Results saved to {csv_filename}")