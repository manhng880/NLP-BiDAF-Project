import torch
import torch.nn as nn
import numpy as np
import json
import os
from model import BiDAF

def word_tokenize(text):
    # Tokenizer đơn giản tương ứng với quá trình preprocess
    return text.lower().replace('.', ' . ').replace(',', ' , ').replace('?', ' ? ').split()

def demo():
    # --- 1. Cấu hình ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = 'save/bidaf_epoch_22.pt' # Sử dụng epoch 22 như đã thảo luận
    hidden_size = 100

    if not os.path.exists(checkpoint_path):
        print(f"❌ Không tìm thấy file checkpoint tại {checkpoint_path}")
        return

    # --- 2. Tải từ điển ---
    print("🔄 Đang nạp từ điển và mô hình...")
    with open('data/processed/word2idx.json', 'r', encoding='utf-8') as f:
        word2idx = json.load(f)
    with open('data/processed/char2idx.json', 'r', encoding='utf-8') as f:
        char2idx = json.load(f)
    
    word_vectors = torch.from_numpy(np.load('data/processed/word_emb.npy')).float()
    char_vocab_size = len(char2idx)

    # --- 3. Khởi tạo và nạp mô hình ---
    model = BiDAF(word_vectors, char_vocab_size, hidden_size).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print("✅ Đã sẵn sàng!\n")

    while True:
        print("-" * 50)
        context = input("Nhập đoạn văn (hoặc 'q' để thoát): ")
        if context.lower() == 'q': break
        
        question = input("Nhập câu hỏi: ")
        if question.lower() == 'q': break

        # --- 4. Tiền xử lý dữ liệu nhập vào ---
        c_tokens = word_tokenize(context)
        q_tokens = word_tokenize(question)

        # Chuyển thành Word Index
        cw = torch.LongTensor([word2idx.get(w, word2idx.get('<UNK>', 1)) for w in c_tokens]).unsqueeze(0).to(device)
        qw = torch.LongTensor([word2idx.get(w, word2idx.get('<UNK>', 1)) for w in q_tokens]).unsqueeze(0).to(device)

        # Chuyển thành Char Index (Max word length = 16)
        def get_char_tensor(tokens):
            ct = torch.zeros(len(tokens), 16).long()
            for i, w in enumerate(tokens):
                for j, c in enumerate(w[:16]):
                    ct[i, j] = char2idx.get(c, char2idx.get('<UNK>', 1))
            return ct.unsqueeze(0).to(device)

        cc = get_char_tensor(c_tokens)
        qc = get_char_tensor(q_tokens)

        # --- 5. Dự đoán ---
        with torch.no_grad():
            p1, p2 = model(cw, cc, qw, qc)
            
            # Lấy vị trí bắt đầu và kết thúc có xác suất cao nhất
            s_idx = torch.argmax(p1, dim=1).item()
            e_idx = torch.argmax(p2, dim=1).item()

            if s_idx <= e_idx:
                answer = " ".join(c_tokens[s_idx : e_idx + 1])
            else:
                answer = "Xin lỗi, tôi không tìm thấy câu trả lời phù hợp."

        print(f"\n🤖 Robot trả lời: {answer.capitalize()}")

if __name__ == "__main__":
    demo()