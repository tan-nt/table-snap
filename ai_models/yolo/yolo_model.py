import locale
import os

from tqdm import tqdm
import multiprocessing
import csv
import os
import json
from PIL import Image
import locale
import torch

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from utils.ted_evaluation import TEDS
from ultralytics import YOLO


header_cells = []
body_cells = []

# Set constants
CONFIDENCE_THRESHOLD = 0.5
ROW_THRESHOLD = 10  # Maximum y_min difference to group cells into the same row
CELL_SPECIAL = ["<b>", "</b>", "<i>", "</i>", "<sup>", "</sup>", "<sub>", "</sub>"]


print("CUDA available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())


os.environ["LC_ALL"] = "C.UTF-8"
os.environ["LANG"] = "C.UTF-8"
locale.setlocale(locale.LC_ALL, "C.UTF-8")


max_processes = multiprocessing.cpu_count()
print('max_processes:', max_processes)


def build_table_from_html_and_cell(
    structure, content
):
    """Build table from html and cell token list"""
    assert structure is not None
    html_code = list()

    # deal with empty table
    if content is None:
        content = ["placeholder"] * len(structure)

    for tag in structure:
        if tag in ("<td>[]</td>", ">[]</td>"):
            if len(content) == 0:
                continue
            cell = content.pop(0)
            html_code.append(tag.replace("[]", cell))
        else:
            html_code.append(tag)

    return html_code

CELL_SPECIAL = ["<b>", "</b>", "<i>", "</i>", "<sup>", "</sup>", "<sub>", "</sub>"]


# Define class mapping for table elements, including spanning cells
class_names_map = {
    # 'table': 0,
    # 'table column': 2,
    # 'table row': 3,
    # 'table projected row header': 5,
    'table cell header': 0,
    'table grid cell': 1,
    # 'table spanning cell': 2, # we use post-processing logic to handle it
}

# Helper function to determine the class of each cell based on structure tokens and cell index
def infer_class_from_structure(structure_tokens, cell_index):
    # anno_code= <thead><tr><td></td><td colspan="7"></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></thead><tbody><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></tbody>
    in_thead = False  # Flag to track header section
    in_tbody = False  # Flag to track body section
    current_cell = 1  # Track index within cells

    for token in structure_tokens:
        # Update section flags based on tags
        if token == "<thead>":
            in_thead = True
        elif token == "</thead>":
            in_thead = False
        elif token == "<tbody>":
            in_tbody = True
        elif token == "</tbody>":
            in_tbody = False

        # Immediately classify as spanning cell if 'colspan' or 'rowspan' is present
        if 'colspan' in token or 'rowspan' in token:
            current_cell += 1
            if current_cell == cell_index:
                # print('cell_index=', cell_index, ', current_cell=', current_cell, ', token=', token, ', in_thead=', in_thead, ", in_tbody=", in_tbody)
                # return class_names_map['table spanning cell']
                return class_names_map['table grid cell']

        # End of a cell and classification
        elif token == "</td>":
            if current_cell == cell_index:
                # print('cell_index=', cell_index, ', current_cell=', current_cell, ', token=', token, ', in_thead=', in_thead, ", in_tbody=", in_tbody)
                # Determine class based on current section
                if in_thead:
                    return class_names_map['table cell header']
                elif in_tbody:
                    return class_names_map['table grid cell']
            # Only increment current_cell after completing <td>...</td> processing
            current_cell += 1
            inside_cell = False  # Reset flag after finishing a cell

    # Default class as table if no specific classification is found
    return class_names_map['table grid cell']


# Helper function to check if two boxes overlap
def boxes_overlap(box1, box2):
    x_min1, y_min1, x_max1, y_max1 = box1
    x_min2, y_min2, x_max2, y_max2 = box2
    return not (x_max1 <= x_min2 or x_min1 >= x_max2 or y_max1 <= y_min2 or y_min1 >= y_max2)

# Filter overlapping boxes based on confidence score
def filter_overlapping_boxes(boxes, confidences):
    filtered_boxes = []
    for i, box1 in enumerate(boxes):
        keep = True
        for j, box2 in enumerate(boxes):
            if i != j and boxes_overlap(box1, box2):
                # Keep the box with the higher confidence score
                if confidences[i] < confidences[j]:
                    keep = False
                    break
        if keep:
            filtered_boxes.append(i)
    return filtered_boxes

def build_table_from_html_and_cell(structure, content):
    """Build table from HTML structure and cell content."""
    assert structure is not None
    html_code = []
    if content is None:
        content = ["placeholder"] * len(structure)
    for tag in structure:
        if tag in ("<td>[]</td>", ">[]</td>"):
            if len(content) == 0:
                continue
            cell = content.pop(0)
            html_code.append(tag.replace("[]", cell))
        else:
            html_code.append(tag)
    return html_code

# Helper function to check if two boxes overlap
def boxes_overlap(box1, box2):
    x_min1, y_min1, x_max1, y_max1 = box1
    x_min2, y_min2, x_max2, y_max2 = box2
    return not (x_max1 <= x_min2 or x_min1 >= x_max2 or y_max1 <= y_min2 or y_min1 >= y_max2)

def filter_overlapping_boxes(boxes, confidences):
    """Filter overlapping boxes based on confidence scores."""
    filtered_boxes = []
    for i, box1 in enumerate(boxes):
        keep = True
        for j, box2 in enumerate(boxes):
            if i != j and boxes_overlap(box1, box2):
                if confidences[i] < confidences[j]:  # Keep higher confidence box
                    keep = False
                    break
        if keep:
            filtered_boxes.append(i)
    return filtered_boxes

def predicted_html_table_template(header_content, body_content):
    """Generate HTML template for predicted table."""
    thead_html = f"<thead>{header_content}</thead>" if header_content else ""
    return f"""<html>
        <head><meta charset="UTF-8">
        <style>
        table, th, td {{
            border: 1px solid black;
            font-size: 10px;
        }}
        </style></head>
        <body>
        <table frame="hsides" rules="groups" width="100%">
            {thead_html}
            <tbody>{body_content}</tbody>
        </table></body></html>"""

def anno_code_html_table_template(table):
    """Generate HTML for groundtruth annotations."""
    return f"""<html>
        <head><meta charset="UTF-8">
        <style>
        table, th, td {{
            border: 1px solid black;
            font-size: 10px;
        }}
        </style></head>
        <body>
        <table frame="hsides" rules="groups" width="100%">
            {table}
        </table></body></html>"""

def calculate_spans(cell, rows, current_row):
    """Calculate rowspan and colspan for a cell."""
    rowspan, colspan = 1, 1
    for next_row in rows:
        for next_cell in next_row:
            if (
                cell["x_min"] == next_cell["x_min"]
                and cell["x_max"] == next_cell["x_max"]
                and next_cell["y_min"] > cell["y_min"]
                and next_cell["y_min"] < cell["y_max"]
            ):
                rowspan += 1
    for next_cell in current_row:
        if (
            cell["y_min"] == next_cell["y_min"]
            and cell["y_max"] == next_cell["y_max"]
            and next_cell["x_min"] > cell["x_min"]
            and next_cell["x_min"] < cell["x_max"]
        ):
            colspan += 1
    return rowspan, colspan


# Additional helper function to resolve overlapping cells within a row
def resolve_row_overlaps(row):
    """
    Resolve overlapping cells within a single row based on x_min.
    Keeps the cell with the higher confidence or larger area in case of ties.
    """
    resolved_row = []
    row.sort(key=lambda cell: cell["x_min"])  # Sort by x_min

    for cell in row:
        if not resolved_row:
            resolved_row.append(cell)
        else:
            last_cell = resolved_row[-1]
            if cell["x_min"] < last_cell["x_max"]:  # Overlap detected
                # Keep the cell with higher confidence
                if cell.get("confidence", 0) > last_cell.get("confidence", 0):
                    resolved_row[-1] = cell
            else:
                resolved_row.append(cell)

    return resolved_row

model = YOLO('ai_models/yolo/yolo_best_weights.pt')

def get_predicted_html_by_yolo(image_path):
    result = model.predict(
        source=image_path,  # Path to your test images
        save=True,              # Save predictions
        conf=CONFIDENCE_THRESHOLD,              # Confidence threshold for predictions
        device="cpu",
        imgsz=640              # Image size, same as during training
    )[0]
    image = Image.open(image_path)
    boxes = result.boxes.xyxy.tolist()
    confidences = result.boxes.conf.tolist()
    class_ids = result.boxes.cls.tolist()
    # Filter overlapping boxes
    filtered_indices = filter_overlapping_boxes(boxes, confidences)
    detected_cells = [
        {
            "class_id": int(class_ids[i]),
            "y_min": boxes[i][1],
            "x_min": boxes[i][0],
            "x_max": boxes[i][2],
            "y_max": boxes[i][3],
        }
        for i in filtered_indices if confidences[i] >= CONFIDENCE_THRESHOLD
    ]
    # Sort and group cells into rows
    detected_cells.sort(key=lambda cell: (cell["y_min"], cell["x_min"]))
    rows, current_row, current_y = [], [], detected_cells[0]["y_min"] if detected_cells else None
    for cell in detected_cells:
        if abs(cell["y_min"] - current_y) > ROW_THRESHOLD:
            current_row = resolve_row_overlaps(current_row)
            rows.append(current_row)
            current_row = []
            current_y = cell["y_min"]
        current_row.append(cell)
    if len(current_row) > 0:
        current_row = resolve_row_overlaps(current_row)
        rows.append(current_row)

    header_html, body_rows_html = "", ""
    processed_cells = set()

    max_cells_row = max(rows, key=len)
    max_columns = len(max_cells_row)
    max_cell_positions = [cell["x_max"] for cell in max_cells_row]  # Get x_min positions of maximum cell row
    max_cell_positions.sort()

    for idx, row in enumerate(rows):
      row_html = ""
      is_header_row = idx == 0 or any(cell["class_id"] == 0 for cell in row)

      max_position_index = 0
      total_filled_cell = 0
      total_cell = 0

      # Sort the row by x_min to ensure cells are processed in the correct horizontal order
      row.sort(key=lambda cell: cell["x_min"])

      for cell in row:
          if (cell["x_min"], cell["y_min"]) in processed_cells:
              continue  # Skip already processed spanning cells

          total_unfilled_cells = 0
           # Fill gaps with missing cells before processing the current cell
          while max_position_index < len(max_cell_positions) and cell["x_min"] > max_cell_positions[max_position_index]:
              max_position_index += 1
              total_unfilled_cells += 1

          total_cell += 1
          if (total_unfilled_cells - total_filled_cell) > 1:
            # Add missing cell with span
            # print('Planning:' , f'<td colspan="{int(total_unfilled_cells)}"></td>')
            row_html += f'<td colspan="{int(total_unfilled_cells)}"></td>'
            total_cell += (total_unfilled_cells - total_filled_cell)
          elif (total_unfilled_cells - total_filled_cell) == 1:
            row_html += '<td></td>'
            # print('Planning:' , '<td></td>')
            total_cell += (total_unfilled_cells - total_filled_cell)
          total_filled_cell += 1

          # Calculate spans for the cell
          rowspan, colspan = calculate_spans(cell, rows, row)
          span_attributes = (
              f' rowspan="{rowspan}"' if rowspan > 1 else ""
          ) + (f' colspan="{colspan}"' if colspan > 1 else "")


          # Add the cell with calculated spans to the row HTML
          row_html += f'<td{span_attributes}></td>'

          # Mark this cell and its spans as processed
          for r in range(rowspan):
              for c in range(colspan):
                  processed_cells.add((cell["x_min"] + c, cell["y_min"] + r))


          # Visualization for spans
          rect = patches.Rectangle(
              (cell["x_min"], cell["y_min"]),
              cell["x_max"] - cell["x_min"],
              cell["y_max"] - cell["y_min"],
              linewidth=2,
              edgecolor="r" if is_header_row else "b",  # Red for headers, blue for body
              facecolor="none",
          )
          max_position_index += 1  # Move to the next max_cell_position

      if total_cell < max_columns:
        # print('max_columns=', max_columns, ', total_cell=', total_cell)
        # Check if there is already a last <td></td> to replace
        if row_html.endswith('<td></td>'):
            # Remove the last <td></td> and replace it with a colspan
            row_html = row_html[:-9] + f'<td colspan="{int(max_columns - total_cell + 1)}"></td>'
        else:
            # Add a new <td> with the calculated colspan if no <td></td> exists
            row_html += f'<td colspan="{int(max_columns - total_cell)}"></td>'
      # print('row_html=', row_html)
      if is_header_row:
          header_html += f"<tr>{row_html}</tr>"
      else:
          body_rows_html += f"<tr>{row_html}</tr>"
    predicted_html = predicted_html_table_template(header_html, body_rows_html)
    return predicted_html


def test_vn_tsr_dataset_by_yolo():
    dataset_folders = 'vn_tsr_dataset'
    teds = TEDS(structure_only=True)
    csv_filename = "/yolo_teds_results.csv"
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
                pred_html = get_predicted_html_by_yolo(img_path)
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

test_vn_tsr_dataset_by_yolo()
