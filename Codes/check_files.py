# check: what files are actually in the folders?
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


from pathlib import Path
from collections import Counter

DATA_DIR = Path(os.path.join(BASE_DIR, "..", "Data", "test"))

for class_name in ["real", "fake"]:
    folder = DATA_DIR / class_name
    all_files = list(folder.iterdir())  # all items without filtering

    # count the extensions
    extensions = Counter(f.suffix.lower() for f in all_files if f.is_file())

    print(f"\n--- folder {class_name} ---")
    print(f"Total items: {len(all_files)}")
    print(f"Extensions: {dict(extensions)}")
