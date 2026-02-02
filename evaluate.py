import torch
import json
import numpy as np
from data_loader import FolkloreDataset
from model import BiDAF
from tqdm import tqdm

def calculate_f1(pred, gold):
    pred_tokens = pred.split()
    gold_tokens = gold.split()
    common = set(pred_tokens) & set(gold_tokens)
    if not common: return 0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * (precision * recall) / (precision + recall)

def evaluate(model_path, data_json, word2idx_path, char2idx_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Dataset
    dataset = FolkloreDataset(data_json, word2idx_path, char2idx_path)
    
    # 2. Load Model
    word_vectors = torch.from_numpy(np.load('data/processed/word_emb.npy')).float()
    char_vocab_size = len(dataset.char2idx)
    model = BiDAF(word_vectors, char_vocab_size=char_vocab_size, hidden_size=100).to(device)
    
    print(f"🔄 Đang nạp trọng số từ: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    total_f1 = 0
    total_em = 0
    results = []

    print(f" Đang đánh giá trên {len(dataset)} mẫu...")
    
    for i in tqdm(range(len(dataset))):
        batch = dataset[i]
        cw = batch['c_word'].unsqueeze(0).to(device)
        cc = batch['c_char'].unsqueeze(0).to(device)
        qw = batch['q_word'].unsqueeze(0).to(device)
        qc = batch['q_char'].unsqueeze(0).to(device)
        y1, y2 = batch['y1'].item(), batch['y2'].item()
        
        with torch.no_grad():
            p1, p2 = model(cw, cc, qw, qc)
            # Lấy vị trí có xác suất cao nhất
            start_pred = torch.argmax(p1, dim=1).item()
            end_pred = torch.argmax(p2, dim=1).item()
        
        # Lấy lại token thực tế
        context_tokens = dataset.data[i]['context_tokens']
        question_text = " ".join(dataset.data[i]['question_tokens'])
        
        pred_text = " ".join(context_tokens[start_pred : end_pred+1])
        gold_text = " ".join(context_tokens[y1 : y2+1])
        
        # Tính điểm
        em = 1 if pred_text == gold_text else 0
        f1 = calculate_f1(pred_text, gold_text)
        
        total_em += em
        total_f1 += f1

    # 3. Hiển thị kết quả
    print("\n" + "="*50)
    print(f" KẾT QUẢ CUỐI CÙNG:")
    print(f" Exact Match (EM): {total_em / len(dataset) * 100:.2f}%")
    print(f" F1 Score: {total_f1 / len(dataset) * 100:.2f}%")
    print("="*50 + "\n")

if __name__ == "__main__":
    # Thay 'bidaf_epoch_20.pt' bằng file bạn muốn test
    evaluate(
        model_path='save/bidaf_epoch_20.pt', 
        data_json='data/dev.json', 
        word2idx_path='data/processed/word2idx.json', 
        char2idx_path='data/processed/char2idx.json'
    )