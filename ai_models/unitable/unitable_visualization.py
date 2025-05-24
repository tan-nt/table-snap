

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns




def visualize_unitable_results():
    # Read CSV data
    df = pd.read_csv('ai_models/unitable/unitable_teds_results.csv')

    # Calculate average TEDS score
    avg_ted_score = df['ted_score'].mean()
    print(f"Average TEDS score: {avg_ted_score:.4f}")

    # 📊 Histogram of TEDS scores
    plt.figure(figsize=(10, 6))
    sns.histplot(df['ted_score'], bins=20, kde=True, color='skyblue')
    plt.axvline(avg_ted_score, color='red', linestyle='--', label=f'Average Score: {avg_ted_score:.2f}')
    plt.title('Distribution of TEDS Scores')
    plt.xlabel('TEDS Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 📉 Lowest 5 TEDS scores
    lowest_5 = df.nsmallest(5, 'ted_score')
    print("\n🔻 Lowest 5 TEDS Score Samples:")
    print(lowest_5[['image_file', 'ted_score']].to_string(index=False))

    plt.figure(figsize=(10, 5))
    sns.barplot(x='ted_score', y='image_file', data=lowest_5, palette='Reds_r')
    plt.title('Lowest 5 TEDS Score Samples')
    plt.xlabel('TEDS Score')
    plt.ylabel('Image File')
    plt.xlim(0, 1)  # TEDS scores between 0 and 1
    plt.show()

    # 📈 Highest 5 TEDS scores
    highest_5 = df.nlargest(5, 'ted_score')
    print("\n🔺 Highest 5 TEDS Score Samples:")
    print(highest_5[['image_file', 'ted_score']].to_string(index=False))

    plt.figure(figsize=(10, 5))
    sns.barplot(x='ted_score', y='image_file', data=highest_5, palette='Greens')
    plt.title('Highest 5 TEDS Score Samples')
    plt.xlabel('TEDS Score')
    plt.ylabel('Image File')
    plt.xlim(0, 1)
    plt.show()

    # 📌 Scatter plot of TEDS scores
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=range(len(df)), y=df['ted_score'], hue=df['ted_score'] < 0.7, palette={True: 'red', False: 'green'})
    plt.title('TEDS Score by Image Index')
    plt.xlabel('Image Index')
    plt.ylabel('TEDS Score')
    plt.grid(True)
    plt.show()