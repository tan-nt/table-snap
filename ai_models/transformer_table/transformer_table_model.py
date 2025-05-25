import os
import csv
from PIL import Image
from utils.ted_evaluation import TEDS
from transformers import DetrFeatureExtractor
from transformers import TableTransformerForObjectDetection
import torch
import pytesseract
import pandas as pd


cell_recognition_model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition")
table_detection_model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
feature_extractor = DetrFeatureExtractor()


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
                # print('box_col',box_col)
                # print('label_col',label_col)
                if label_col == 1:
                    cell_box = (box_col[0], box_row[1], box_col[2], box_row[3])
                    cell_locations.append(cell_box)

    cell_locations.sort(key=lambda x: (x[1], x[0]))
    if len(cell_locations) == 0:
        return None

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


def test_vn_tsr_dataset_by_transformer_table():
    dataset_folders = 'vn_tsr_dataset'
    teds = TEDS(structure_only=True)
    csv_filename = "ai_models/transformer_table/transformer_table_teds_results.csv"
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
        for dataset_folder in os.listdir(dataset_folders):
            if not os.path.isdir(f'{dataset_folders}/{dataset_folder}'):
                continue
            if dataset_folder not in ['digital', 'printed']:
                continue

            for folder_data in os.listdir(f'{dataset_folders}/{dataset_folder}'):
                img_path = f'{dataset_folders}/{dataset_folder}/{folder_data}/img/{folder_data}.png'
                if 'table' not in img_path:
                    continue
                if img_path in existing_results:
                    print(f"Skipping already processed file: {img_path}")
                    continue

                table = extract_table(img_path)
                if table is None:
                    continue
                pred_html = table.to_html()
                anno_html = open(f'{dataset_folders}/{dataset_folder}/{folder_data}/annotation/content.html', 'r', encoding='utf-8').read()
                score = teds.evaluate(pred_html, anno_html)
                print(f"TEDS Score: {score}, Index: {index}, Image Path: {img_path}")
                index += 1

                # Save to CSV immediately
                writer.writerow({
                    "image_file": img_path,
                    "ted_score": score,
                    "pred_html": pred_html,
                    "annotation_html": anno_html
                })
                csv_file.flush()  # Force write to disk after each entry

    print(f"Results saved to {csv_filename}")