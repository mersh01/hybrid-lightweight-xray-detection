import cv2
import os
import shutil
from pathlib import Path
from tqdm import tqdm

def convert_dataset_local():
    source_dir = Path(r"c:\Users\merao\OneDrive\Documents\Reasearch\HiXray")
    target_dir = Path(r"c:\Users\merao\OneDrive\Documents\Reasearch\HiXray_YOLO")
    
    CLASSES = [
        'Cosmetic', 'Laptop', 'Mobile_Phone', 'Nonmetallic_Lighter', 
        'Portable_Charger_1', 'Portable_Charger_2', 'Tablet', 'Water'
    ]
    class_to_id = {cls_name: i for i, cls_name in enumerate(CLASSES)}
    
    splits = [('train', 'train'), ('test', 'val')]
    
    # Create target directories
    for _, yolo_split in splits:
        (target_dir / 'images' / yolo_split).mkdir(parents=True, exist_ok=True)
        (target_dir / 'labels' / yolo_split).mkdir(parents=True, exist_ok=True)
        
    for original_split, yolo_split in splits:
        img_dir = source_dir / original_split / f"{original_split}_image"
        ann_dir = source_dir / original_split / f"{original_split}_annotation"
        
        if not img_dir.exists() or not ann_dir.exists():
            print(f"Warning: Missing directories for {original_split}. Skipping...")
            continue
            
        print(f"Processing {original_split} split into {yolo_split}...")
        ann_files = list(ann_dir.glob('*.txt'))
        
        for ann_file in tqdm(ann_files):
            with open(ann_file, 'r') as f:
                lines = f.readlines()
                
            if not lines:
                continue
                
            # Assume image name matches annotation stem
            img_path = img_dir / f"{ann_file.stem}.jpg"
            if not img_path.exists():
                # Try getting name from first line if direct match fails
                img_name = lines[0].strip().split(' ')[0]
                img_path = img_dir / img_name
                if not img_path.exists():
                    continue
                    
            # Read image to get dims
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h_img, w_img = img.shape[:2]
            
            yolo_lines = []
            valid_labels = False
            for line in lines:
                parts = line.strip().split(' ')
                if len(parts) < 6: continue
                _, cls_name, xmin, ymin, xmax, ymax = parts[:6]
                
                if cls_name not in class_to_id: continue
                cls_id = class_to_id[cls_name]
                
                xmin, ymin, xmax, ymax = float(xmin), float(ymin), float(xmax), float(ymax)
                
                x_center = ((xmin + xmax) / 2.0) / w_img
                y_center = ((ymin + ymax) / 2.0) / h_img
                width = (xmax - xmin) / w_img
                height = (ymax - ymin) / h_img
                
                x_center, y_center = max(0, min(1, x_center)), max(0, min(1, y_center))
                width, height = max(0, min(1, width)), max(0, min(1, height))
                
                yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                valid_labels = True
                
            if valid_labels:
                # Copy image
                target_img_path = target_dir / 'images' / yolo_split / img_path.name
                shutil.copy2(img_path, target_img_path)
                
                # Write label
                target_label_path = target_dir / 'labels' / yolo_split / f"{ann_file.stem}.txt"
                with open(target_label_path, 'w') as out_f:
                    out_f.write('\\n'.join(yolo_lines))

    # Generate dataset.yaml
    yaml_content = f"""
path: {target_dir.as_posix()}
train: images/train
val: images/val

names:
  0: Cosmetic
  1: Laptop
  2: Mobile_Phone
  3: Nonmetallic_Lighter
  4: Portable_Charger_1
  5: Portable_Charger_2
  6: Tablet
  7: Water
"""
    with open(target_dir / 'dataset.yaml', 'w') as f:
        f.write(yaml_content)
        
    print("Local conversion complete!")

if __name__ == "__main__":
    convert_dataset_local()
