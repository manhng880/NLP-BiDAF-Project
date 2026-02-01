import json
import random
import os

def split_data(input_file, train_file, dev_file, split_ratio=0.8):
    if not os.path.exists(input_file):
        print(f" Không tìm thấy file {input_file}")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f) # data bây giờ là một list [{}, {}, ...]

    # Shuffle dữ liệu để đảm bảo tính ngẫu nhiên
    random.seed(42) # Giữ cố định để nếu chạy lại kết quả split không đổi
    random.shuffle(data)

    split_index = int(len(data) * split_ratio)
    
    train_data = data[:split_index]
    dev_data = data[split_index:]

    # Lưu file Train
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)
    
    # Lưu file Dev
    with open(dev_file, 'w', encoding='utf-8') as f:
        json.dump(dev_data, f, ensure_ascii=False, indent=4)

    print(f" Đã chia dữ liệu thành công!")
    print(f" - Tổng cộng: {len(data)} QA")
    print(f" - Tập Train (80%): {len(train_data)} QA")
    print(f" - Tập Dev (20%): {len(dev_data)} QA")

if __name__ == "__main__":
    # Sử dụng file output từ preprocess.py làm đầu vào cho split
    split_data(
        "vietnamese_folktale_char_preprocessed.json", 
        "data/train.json", 
        "data/dev.json"
    )