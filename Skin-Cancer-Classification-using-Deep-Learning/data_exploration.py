import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import sys # Add sys import for error handling

# Define the expected CSV path
csv_path = 'Data/Processed CSV\'s/train_2020_and_2019_withPateintDetail_9_labels.csv'
# Define the expected image directory structure (adjust if needed based on actual data download)
# Example: Using 512x512 images relative to the project root
image_base_dir = '512x512' # Adjust this if images are elsewhere or named differently

# Step 1: Load the CSV
print(f"Attempting to load CSV from: {csv_path}")
if not os.path.exists(csv_path):
    print(f"Error: CSV file not found at {csv_path}", file=sys.stderr)
    print("Please ensure the dataset CSV is available at the specified path.", file=sys.stderr)
    sys.exit(1)

try:
    df = pd.read_csv(csv_path)
except Exception as e:
    print(f"Error loading CSV: {e}", file=sys.stderr)
    sys.exit(1)

print("CSV loaded successfully.")

# Step 2: Check the structure of the dataset
print("\n--- DataFrame Head ---")
print(df.head())
print("\n--- DataFrame Info ---")
df.info()
print("\n--- DataFrame Columns ---")
print(df.columns)

# Step 3: Check class distribution
diagnosis_col = 'diagnosis' # Adjust if the column name is different (e.g., 'target', 'benign_malignant')
if diagnosis_col not in df.columns:
    # Try finding a likely target column if 'diagnosis' is missing
    potential_cols = ['target', 'benign_malignant', 'label', 'melanoma']
    found_col = None
    for col in potential_cols:
        if col in df.columns:
            diagnosis_col = col
            print(f"Warning: '{diagnosis_col}' column not found. Using '{col}' instead.")
            found_col = True
            break
    if not found_col:
         print(f"Error: Cannot find '{diagnosis_col}' or alternative target column in the DataFrame.", file=sys.stderr)
         sys.exit(1)


print(f"\n--- Value Counts for '{diagnosis_col}' ---")
print(df[diagnosis_col].value_counts())

print("\nGenerating class distribution plot...")
plt.figure(figsize=(10, 6)) # Adjust figure size for better readability
sns.countplot(data=df, x=diagnosis_col)
plt.xticks(rotation=60, ha='right') # Improve rotation and alignment
plt.title('Class Distribution')
plt.tight_layout() # Adjust layout to prevent labels overlapping
plot_filename = 'class_distribution.png'
plt.savefig(plot_filename)
print(f"Class distribution plot saved to {plot_filename}")
# plt.show() # Avoid showing interactively, save instead

# Step 4: Attach full image path (Revised)
# Check if an image path column already exists from preprocessing (like the one added by utils.append_path)
path_col = 'image_path' # A potential pre-existing path column
image_name_col = 'image' # Expected column with image ID/name

if path_col not in df.columns:
    print(f"Column '{path_col}' not found. Attempting to create it.")
    if image_name_col not in df.columns:
         print(f"Error: Cannot find required column '{image_name_col}' to build image paths.", file=sys.stderr)
         sys.exit(1)

    # Check if the base image directory exists
    if not os.path.isdir(image_base_dir):
         print(f"Warning: Image directory '{image_base_dir}' not found. Image paths may be invalid.", file=sys.stderr)
         print(f"Please ensure images are downloaded and located at '{image_base_dir}'.", file=sys.stderr)

    print(f"Creating '{path_col}' using base directory '{image_base_dir}' and column '{image_name_col}'.")
    # Assuming image names in the CSV do not have extensions
    df[path_col] = df[image_name_col].apply(lambda x: os.path.join(image_base_dir, f'{x}.jpg'))
    print(f"Created '{path_col}'. Sample:")
    print(df[[image_name_col, path_col]].head())
else:
     print(f"Using existing column '{path_col}' for image paths.")
     # Optional: Verify a few paths exist
     if not df.empty and path_col in df.columns:
        sample_path = df[path_col].iloc[0]
        if not os.path.exists(sample_path):
             print(f"Warning: Sample image path '{sample_path}' does not exist.", file=sys.stderr)
             print("Please check if the image paths in the CSV are correct and images are downloaded.", file=sys.stderr)


# Step 5: Train-validation split
print("\nSplitting data into training and validation sets...")
try:
    train_df, val_df = train_test_split(df,
                                        test_size=0.2,
                                        stratify=df[diagnosis_col],
                                        random_state=42)
    print("Data split successfully.")
    print(f"Training set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")

    # Optional: Save the split dataframes for later use
    # train_df.to_csv('train_split.csv', index=False)
    # val_df.to_csv('validation_split.csv', index=False)
    # print("Saved train_split.csv and validation_split.csv")

except Exception as e:
    print(f"Error during train-test split: {e}", file=sys.stderr)
    sys.exit(1)

print("\nScript finished. Ready to use train_df and val_df.") 