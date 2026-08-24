"""
Dataset preparation script for PlantDoc dataset.
Maps raw PlantDoc folder names to our 15 PlantVillage target classes and builds structured data/plantdoc/ train and test splits.
"""

import os
import shutil

PLANTDOC_MAPPING = {
    "Bell_pepper leaf spot": "Pepper__bell___Bacterial_spot",
    "Bell_pepper leaf": "Pepper__bell___healthy",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Potato leaf": "Potato___healthy",
    "Tomato leaf bacterial spot": "Tomato_Bacterial_spot",
    "Tomato Early blight leaf": "Tomato_Early_blight",
    "Tomato leaf late blight": "Tomato_Late_blight",
    "Tomato mold leaf": "Tomato_Leaf_Mold",
    "Tomato Septoria leaf spot": "Tomato_Septoria_leaf_spot",
    "Tomato two spotted spider mites leaf": "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato target spot leaf": "Tomato__Target_Spot",
    "Tomato leaf yellow virus": "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato leaf mosaic virus": "Tomato__Tomato_mosaic_virus",
    "Tomato leaf": "Tomato_healthy"
}

def setup_plantdoc(raw_dir="data/plantdoc_raw", target_dir="data/plantdoc"):
    print("Setting up PlantDoc dataset splits...")
    os.makedirs(target_dir, exist_ok=True)

    for split in ["train", "test"]:
        raw_split_dir = os.path.join(raw_dir, split)
        target_split_dir = os.path.join(target_dir, split)
        os.makedirs(target_split_dir, exist_ok=True)

        if not os.path.exists(raw_split_dir):
            continue

        for folder_name in os.listdir(raw_split_dir):
            if folder_name not in PLANTDOC_MAPPING:
                continue

            pv_class_name = PLANTDOC_MAPPING[folder_name]
            src_class_path = os.path.join(raw_split_dir, folder_name)
            dst_class_path = os.path.join(target_split_dir, pv_class_name)

            os.makedirs(dst_class_path, exist_ok=True)

            img_count = 0
            for img_name in os.listdir(src_class_path):
                src_img = os.path.join(src_class_path, img_name)
                dst_img = os.path.join(dst_class_path, img_name)
                if os.path.isfile(src_img):
                    shutil.copy2(src_img, dst_img)
                    img_count += 1

            print(f"Copied {img_count} images for split '{split}' -> {pv_class_name}")

    print("PlantDoc dataset setup completed successfully!")

if __name__ == "__main__":
    setup_plantdoc()
