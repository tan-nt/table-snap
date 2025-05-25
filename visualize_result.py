import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image
import random


def save_teds_score():
    result_files = [
        {
            'name': 'unitable',
            'path': 'ai_models/unitable/unitable_teds_results.csv'
        },
        {
            'name': 'yolo',
            'path': 'ai_models/yolo/yolo_teds_results.csv'
        },
        {
            'name': 'table_transformer',
            'path': 'ai_models/transformer_table/transformer_table_teds_results.csv'
        },
        {
            'name': 'contours_detection',
            'path': 'ai_models/contours_detection/contours_detection_teds_results.csv'
        }
    ]
    teds_score_list = []
    for result_file in result_files:
        df = pd.read_csv(result_file['path'])
        teds_score = df['ted_score'].mean()
        print('model=', result_file['name'], 'teds_score=', teds_score)
        teds_score_list.append({
            'name': result_file['name'],
            'ted_score': teds_score
        })
    print('teds_score_list=', teds_score_list)
    teds_score_df = pd.DataFrame(teds_score_list)
    teds_score_df.to_csv('teds_score.csv', index=False)


def draw_teds_score_chart():
    teds_score_df = pd.read_csv('teds_score.csv')

    teds_score_df = teds_score_df.sort_values(by='ted_score', ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(teds_score_df['name'], teds_score_df['ted_score'], color='skyblue')
    plt.xlabel('TEDS Score')
    plt.ylabel('Model')
    plt.title('TEDS Score Comparison of Models on Vietnamese TSR Dataset')
    plt.xlim(0, 1)
    for index, value in enumerate(teds_score_df['ted_score']):
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()

    # Save chart image for report
    plt.savefig('teds_score_comparison_chart.png')
    plt.show()


def visualize_validation_results():
    # Read CSV data
    ted_result_files = [
        {
            "name": "Unitable",
            "path": "ai_models/unitable/unitable_teds_results.csv"
        },
        {
            "name": "Table Transformer",
            "path": "ai_models/transformer_table/transformer_table_teds_results.csv"
        }
    ]

    for ted_result_file in ted_result_files:
        df = pd.read_csv(ted_result_file['path'])
        avg_ted_score = df['ted_score'].mean()
        print(f"Average TEDS score: {avg_ted_score:.4f}")

        plt.figure(figsize=(10, 6))
        sns.histplot(df['ted_score'], bins=20, kde=True, color='skyblue')
        plt.axvline(avg_ted_score, color='red', linestyle='--', label=f'Average Score: {avg_ted_score:.2f}')
        plt.title(f'Distribution of TEDS Scores in {ted_result_file["name"]}')
        plt.xlabel('TEDS Score')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.show()

        df['image_filename'] = df['image_file'].apply(os.path.basename)
        lowest_5 = (
            df.drop_duplicates(subset=['ted_score'])
            .nsmallest(5, 'ted_score')
        )
        print(f"\n🔻 Lowest 5 TEDS Score Samples in {ted_result_file['name']}:")
        print(lowest_5[['image_filename', 'ted_score']].to_string(index=False))

        plt.figure(figsize=(12, 8))  # Make figure bigger
        sns.barplot(x='ted_score', y='image_filename', data=lowest_5, palette='Reds_r')
        plt.title(f'Lowest 5 TEDS Score Samples in {ted_result_file["name"]}')
        plt.xlabel('TEDS Score')
        plt.ylabel('Image File')
        plt.xlim(0, 1)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()
        plt.show()

        highest_5 = (
            df.drop_duplicates(subset=['ted_score'])
            .nlargest(5, 'ted_score')
        )
        print(f"\n🔺 Highest 5 TEDS Score Samples in {ted_result_file['name']}:")
        print(highest_5[['image_filename', 'ted_score']].to_string(index=False))

        plt.figure(figsize=(12, 8))
        sns.barplot(x='ted_score', y='image_filename', data=highest_5, palette='Greens')
        plt.title(f'Highest 5 TEDS Score Samples in {ted_result_file["name"]}')
        plt.xlabel('TEDS Score')
        plt.ylabel('Image File')
        plt.xlim(0, 1)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=range(len(df)), y=df['ted_score'], hue=df['ted_score'] >= 0.7, palette={True: 'green', False: 'red'})
        plt.title(f'TEDS Score by Image Index in {ted_result_file["name"]}')
        plt.xlabel('Image Index')
        plt.ylabel('TEDS Score')
        plt.grid(True)
        plt.show()


def display_lowest_highest_teds_images(df, ted_result_file):
    # 📉 Lowest 5 TEDS scores
    lowest_5 = (
        df.drop_duplicates(subset=['ted_score'])
        .nsmallest(5, 'ted_score')
    )
    print(f"\n🔻 Lowest 5 TEDS Score Samples in {ted_result_file['name']}:")
    print(lowest_5[['image_file', 'ted_score']].to_string(index=False))

    # Display lowest 5 images
    plt.figure(figsize=(15, 5))
    for i, row in enumerate(lowest_5.itertuples()):
        img = Image.open(row.image_file)
        plt.subplot(1, 5, i+1)
        plt.imshow(img)
        plt.title(f'{row.ted_score:.3f}')
        plt.axis('off')
    plt.suptitle(f'Lowest 5 TEDS Score Samples in {ted_result_file["name"]}')
    plt.show()

    # 📈 Highest 5 TEDS scores
    highest_5 = (
        df.drop_duplicates(subset=['ted_score'])
        .nlargest(5, 'ted_score')
    )
    print(f"\n🔺 Highest 5 TEDS Score Samples in {ted_result_file['name']}:")
    print(highest_5[['image_file', 'ted_score']].to_string(index=False))

    # Display highest 5 images
    plt.figure(figsize=(15, 5))
    for i, row in enumerate(highest_5.itertuples()):
        img = Image.open(row.image_file)
        plt.subplot(1, 5, i+1)
        plt.imshow(img)
        plt.title(f'{row.ted_score:.3f}')
        plt.axis('off')
    plt.suptitle(f'Highest 5 TEDS Score Samples in {ted_result_file["name"]}')
    plt.show()


def get_random_numbers():
    return [random.randint(100, 500) for _ in range(5)]


def visualize_transformed_images():
    dataset_folders = 'vn_tsr_dataset'

    for dataset_folder in os.listdir(dataset_folders):
        if not os.path.isdir(f'{dataset_folders}/{dataset_folder}'):
            continue
        if dataset_folder not in ['digital', 'printed']:
            continue

        random_numbers = get_random_numbers()
        img_paths = []
        for random_number in random_numbers:
            six_digit_str = f'{random_number:06d}'  # Format as 6 digits
            img_path = f'{dataset_folders}/{dataset_folder}/{dataset_folder}_table_{six_digit_str}/img/{dataset_folder}_table_{six_digit_str}.png'
            if os.path.exists(img_path):
                img_paths.append(img_path)

        # Display the images
        plt.figure(figsize=(15, 5))
        for i, img_path in enumerate(img_paths):
            img = Image.open(img_path)
            plt.subplot(1, len(img_paths), i + 1)
            plt.imshow(img)
            plt.title(os.path.basename(img_path))
            plt.axis('off')
        plt.suptitle(f'Random Samples from {dataset_folder}')
        plt.show()


def visualize_lowest_highest_teds_images():
    df = pd.read_csv('ai_models/unitable/unitable_teds_results.csv')
    display_lowest_highest_teds_images(df, {'name': 'Unitable'})

    df = pd.read_csv('ai_models/transformer_table/transformer_table_teds_results.csv')
    display_lowest_highest_teds_images(df, {'name': 'Table Transformer'})


# save_teds_score()
# draw_teds_score_chart()
# visualize_validation_results()
# visualize_lowest_highest_teds_images()
visualize_transformed_images()