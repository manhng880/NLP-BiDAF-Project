import json
import numpy as np
from tqdm import tqdm
import os

# Cấu hình đường dẫn cho đúng với cấu hình sau khi split
RAW_VEC_PATH = ".vector_cache/cc.vi.300.vec"
TRAIN_FILE = "data/train.json"
DEV_FILE = "data/dev.json"
OUT_DIR = "data/processed"

def create_embeddings():
    words = set()
    chars = set()
    
    # 1. Thu thập từ và ký tự từ CẢ HAI tập Train và Dev
    for fpath in [TRAIN_FILE, DEV_FILE]:
        if not os.path.exists(fpath):
            print(f" Không tìm thấy file {fpath}. Hãy chạy split_data.py trước!")
            return
            
        with open(fpath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                all_tokens = item['context_tokens'] + item['question_tokens']
                words.update(all_tokens)
                for token in all_tokens:
                    chars.update(list(token))

    # 2. Load FastText vào RAM (Chỉ lấy những từ có trong bộ dữ liệu của mình)
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    embeddings = [np.zeros(300), np.random.uniform(-0.1, 0.1, 300)]
    
    full_fasttext = {}
    print(f" Đang nạp FastText để lọc ra {len(words)} từ...")
    
    # Lưu ý: File .vec của FastText thường rất nặng, tqdm giúp theo dõi tiến độ
    with open(RAW_VEC_PATH, 'r', encoding='utf-8', errors='ignore') as f:
        next(f) # Bỏ qua dòng header
        for line in tqdm(f, desc="Loading FastText"):
            parts = line.rstrip().split(' ')
            word = parts[0]
            # Kiểm tra xem từ trong FastText có nằm trong tập từ của mình không
            if word in words or word.lower() in words:
                full_fasttext[word] = np.array([float(x) for x in parts[1:]])

    # 3. Xây dựng ma trận Embedding Word
    found = 0
    for word in words:
        # Ưu tiên từ gốc, sau đó thử với lowercase
        w_lookup = word if word in full_fasttext else word.lower()
        
        if w_lookup in full_fasttext:
            word2idx[word] = len(embeddings)
            embeddings.append(full_fasttext[w_lookup])
            found += 1
        else:
            word2idx[word] = 1 # Gán là UNK nếu không tìm thấy trong FastText

    # 4. Tạo char2idx (Cực kỳ quan trọng cho Char Embedding)
    char2idx = {"<PAD>": 0, "<UNK>": 1}
    for c in sorted(list(chars)):
        char2idx[c] = len(char2idx)

    # 5. Lưu tất cả vào thư mục data/processed
    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(os.path.join(OUT_DIR, "word_emb.npy"), np.array(embeddings))
    
    with open(os.path.join(OUT_DIR, "word2idx.json"), "w", encoding='utf-8') as f:
        json.dump(word2idx, f, ensure_ascii=False, indent=4)
        
    with open(os.path.join(OUT_DIR, "char2idx.json"), "w", encoding='utf-8') as f:
        json.dump(char2idx, f, ensure_ascii=False, indent=4)

    print(f"\n Đã chuẩn bị xong dữ liệu huấn luyện!")
    print(f" Tỷ lệ phủ Word: {found}/{len(words)} ({found/len(words)*100:.2f}%)")
    print(f" Số lượng ký tự trong từ điển: {len(char2idx)}")
    print(f" Kết quả lưu tại: {OUT_DIR}")

if __name__ == "__main__":
    create_embeddings()