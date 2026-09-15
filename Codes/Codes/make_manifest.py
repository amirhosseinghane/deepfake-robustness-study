# Step 2: build the manifest
# This code finds all images and creates a CSV file
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


from pathlib import Path   # for safely working with paths
import pandas as pd        # for building the table and saving the CSV

# --- Data folder path (set here according to your system) ---
DATA_DIR = Path(os.path.join(BASE_DIR, "..", "Data", "test"))

# Output folder for saving the manifest
OUTPUT_DIR = Path(BASE_DIR)

# --- Define classes: folder name and its numeric label ---
# real = 0, fake = 1
classes = {"real": 0, "fake": 1}

rows = []  # collect all rows here

# --- Iterate over each folder ---
for class_name, label in classes.items():
    folder = DATA_DIR / class_name          # path to this class's folder
    images = list(folder.glob("*.png")) + list(folder.glob("*.jpg"))
    print(f"Folder {class_name}: {len(images)} images found")

    for img_path in images:
        rows.append({
            "filepath": str(img_path),  # full file path
            "label": label,             # 0 or 1
            "class_name": class_name,   # class name (for readability)
            "split": "test"             # this data is from the test split
        })

# --- Build the table and save it ---
df = pd.DataFrame(rows)

output_path = OUTPUT_DIR / "manifest.csv"
df.to_csv(output_path, index=False, encoding="utf-8")

# --- Final summary for review ---
print("\n--- Summary ---")
print(f"Total images: {len(df)}")
print(f"Count per class:\n{df['class_name'].value_counts()}")
print(f"\nManifest saved at: {output_path}")
