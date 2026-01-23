import torch
import numpy as np
import json
import os
from model import BiDAF

def predict(context_text, question_text, model_path, word2idx_path):
    device = torch.device('cpu')
    
    # 1. Load từ điển
    with open(word2idx_path, 'r', encoding='utf-8') as f:
        word2idx = json.load(f)
    
    # 2. Load Model
    word_vectors = torch.from_numpy(np.load('data/processed/word_emb.npy')).float()
    model = BiDAF(word_vectors=word_vectors, char_vocab_size=100, emb_dim=300, hidden_size=100)
    
    if not os.path.exists(model_path):
        return "Lỗi: Không tìm thấy file model đã huấn luyện!"
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3. Tiền xử lý
    c_tokens = context_text.lower().split()
    q_tokens = question_text.lower().split()
    
    # Padding/Truncate context về đúng max_len (600 như trong data_loader)
    c_idx_list = [word2idx.get(w, 1) for w in c_tokens[:600]]
    if len(c_idx_list) < 600:
        c_idx_list += [0] * (600 - len(c_idx_list))
        
    q_idx_list = [word2idx.get(w, 1) for w in q_tokens[:50]]
    if len(q_idx_list) < 50:
        q_idx_list += [0] * (50 - len(q_idx_list))

    c_idx = torch.tensor(c_idx_list).unsqueeze(0)
    q_idx = torch.tensor(q_idx_list).unsqueeze(0)

    # 4. Dự đoán
    with torch.no_grad():
        p1, p2 = model(c_idx, q_idx) # Output: [1, 600]
        
        # Áp dụng Softmax để lấy xác suất
        p1 = torch.softmax(p1, dim=1)
        p2 = torch.softmax(p2, dim=1)
        
        # Tìm cặp (i, j) sao cho i <= j và p1[i]*p2[j] lớn nhất
        max_prob = 0
        start_idx = 0
        end_idx = 0
        
        # Thuật toán tìm kiếm tối ưu cho QA
        for i in range(len(c_tokens[:600])):
            for j in range(i, min(i + 15, len(c_tokens[:600]))): # Giả sử câu trả lời không quá 15 từ
                prob = p1[0, i] * p2[0, j]
                if prob > max_prob:
                    max_prob = prob
                    start_idx = i
                    end_idx = j

    # 5. Trích xuất
    prediction = " ".join(c_tokens[start_idx : end_idx + 1])
    return prediction, start_idx, end_idx

if __name__ == "__main__":
    # Tự động lấy dữ liệu từ dev.json để test
    with open('data/dev.json', 'r', encoding='utf-8') as f:
        dev_data = json.load(f)['data']

    # Lấy mẫu đầu tiên
    story = dev_data[2]['paragraphs'][0]
    context = story['context']
    qa = story['qas'][0]
    question = qa['question']
    target = qa['answers'][0]['text']

    print(f"❓ Câu hỏi: {question}")
    print(f"🎯 Đáp án đúng: {target}")

    # Sử dụng file model tốt nhất
    ans, s, e = predict(context, question, 'save/bidaf_best.pt', 'data/processed/word2idx.json')

    print(f"🤖 BiDAF dự đoán (từ {s} đến {e}): {ans}")