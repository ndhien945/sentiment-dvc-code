import os
import json
import random

def aggregate_and_split(data_root, dataset_type="UIT_HWDB_line", val_ratio=0.1, seed=42):
    """
    Gom cụm dữ liệu từ các thư mục con và chia tập Train/Val.
    """
    random.seed(seed)
    
    # Tạo thư mục chứa dữ liệu đã xử lý nếu chưa có
    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    # 1. Xử lý tập Train & Validation
    train_raw_dir = os.path.join(data_root, dataset_type, dataset_type, "train_data")
    all_train_samples = []
    
    print(f"Đang quét dữ liệu huấn luyện tại: {train_raw_dir}")
    if os.path.exists(train_raw_dir):
        for folder_id in os.listdir(train_raw_dir):
            folder_path = os.path.join(train_raw_dir, folder_id)
            if os.path.isdir(folder_path):
                label_path = os.path.join(folder_path, "label.json")
                if os.path.exists(label_path):
                    with open(label_path, 'r', encoding='utf-8') as f:
                        labels = json.load(f)
                        
                    for img_name, text in labels.items():
                        # Lưu đường dẫn tương đối để code linh hoạt trên nhiều môi trường (Kaggle/Local)
                        img_rel_path = os.path.join(folder_path, img_name)
                        all_train_samples.append({
                            "image": img_rel_path,
                            "text": text.strip()
                        })
    
    # Xáo trộn và chia tách
    random.shuffle(all_train_samples)
    val_size = int(len(all_train_samples) * val_ratio)
    val_samples = all_train_samples[:val_size]
    train_samples = all_train_samples[val_size:]
    
    print(f"Tổng số mẫu Train gốc: {len(all_train_samples)}")
    print(f"-> Đã chia thành: {len(train_samples)} Train và {len(val_samples)} Validation.")

    # 2. Xử lý tập Test (Chỉ gom cụm, không chia)
    test_raw_dir = os.path.join(data_root, dataset_type, dataset_type, "test_data")
    test_samples = []
    
    print(f"\nĐang quét dữ liệu kiểm thử tại: {test_raw_dir}")
    if os.path.exists(test_raw_dir):
        for folder_id in os.listdir(test_raw_dir):
            folder_path = os.path.join(test_raw_dir, folder_id)
            if os.path.isdir(folder_path):
                label_path = os.path.join(folder_path, "label.json")
                if os.path.exists(label_path):
                    with open(label_path, 'r', encoding='utf-8') as f:
                        labels = json.load(f)
                        
                    for img_name, text in labels.items():
                        img_rel_path = os.path.join(folder_path, img_name)
                        test_samples.append({
                            "image": img_rel_path,
                            "text": text.strip()
                        })
    print(f"Tổng số mẫu Test: {len(test_samples)}")

    # 3. Lưu ra file JSON tổng hợp
    train_out = os.path.join(processed_dir, "train_split.json")
    val_out = os.path.join(processed_dir, "val_split.json")
    test_out = os.path.join(processed_dir, "test_split.json")
    
    with open(train_out, 'w', encoding='utf-8') as f:
        json.dump(train_samples, f, ensure_ascii=False, indent=4)
    with open(val_out, 'w', encoding='utf-8') as f:
        json.dump(val_samples, f, ensure_ascii=False, indent=4)
    with open(test_out, 'w', encoding='utf-8') as f:
        json.dump(test_samples, f, ensure_ascii=False, indent=4)
        
    print(f"\n[Hoàn tất] Đã lưu các file index tại: {processed_dir}")

if __name__ == "__main__":
    # Thay đổi đường dẫn data_root cho phù hợp với máy của bạn hoặc Kaggle
    data_root = "data" 
    aggregate_and_split(data_root=data_root, dataset_type="UIT_HWDB_line", val_ratio=0.1)