# -*- coding: utf-8 -*-
# بررسی: چه فایل‌هایی واقعاً در پوشه‌ها هستند؟
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


from pathlib import Path
from collections import Counter

DATA_DIR = Path(os.path.join(BASE_DIR, "..", "Data", "test"))

for class_name in ["real", "fake"]:
    folder = DATA_DIR / class_name
    all_files = list(folder.iterdir())  # همه‌ی آیتم‌ها بدون فیلتر

    # شمردن پسوندها
    extensions = Counter(f.suffix.lower() for f in all_files if f.is_file())

    print(f"\n--- پوشه‌ی {class_name} ---")
    print(f"مجموع آیتم‌ها: {len(all_files)}")
    print(f"پسوندها: {dict(extensions)}")