import json
import random

def split_again(input_file, train_file, dev_file, split_ratio=0.8):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_stories = data['data']
    # Sử dụng random.seed nếu bạn muốn kết quả split giống hệt lần trước (nếu cần)
    # random.seed(42) 
    random.shuffle(all_stories)

    split_index = int(len(all_stories) * split_ratio)
    
    train_stories = all_stories[:split_index]
    dev_stories = all_stories[split_index:]

    # Lưu file Train mới
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump({"version": "v1.0", "data": train_stories}, f, ensure_ascii=False, indent=4)
    
    # Lưu file Dev mới
    with open(dev_file, 'w', encoding='utf-8') as f:
        json.dump({"version": "v1.0", "data": dev_stories}, f, ensure_ascii=False, indent=4)

    print(f"✅ Đã split lại thành công!")
    print(f" - Train: {len(train_stories)} truyện với ID mới")
    print(f" - Dev: {len(dev_stories)} truyện với ID mới")

# Thực hiện split
split_again("vietnamese_folktale_50_final_v2.json", "train.json", "dev.json")