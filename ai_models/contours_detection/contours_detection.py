import os
import csv
from utils.ted_evaluation import TEDS
import cv2
from app.table_extraction.table_extraction import detect_and_show_table, extract_table_from_image


def detect_pred_html(file):
    table_detected_image, contours = detect_and_show_table(file)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea) # Process the largest table
        table = extract_table_from_image(file, largest_contour)
        return table.to_html()
    return None


def test_vn_tsr_dataset_by_contours_detection():
    dataset_folders = 'vn_tsr_dataset'
    teds = TEDS(structure_only=True)
    csv_filename = "ai_models/contours_detection/contours_detection_teds_results.csv"
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
                if img_path in existing_results:
                    print(f"Skipping already processed file: {img_path}")
                    continue

                if 'table' not in img_path:
                    continue

                anno_html = open(f'{dataset_folders}/{dataset_folder}/{folder_data}/annotation/content.html', 'r', encoding='utf-8').read()
                pred_html = detect_pred_html(img_path)
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
