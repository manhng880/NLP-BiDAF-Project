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
    if len(common) == 0: return 0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * (precision * recall) / (precision + recall)

def evaluate(model_path, data_json, word2idx_path):
    device = torch.device('cpu')
    dataset = FolkloreDataset(data_json, word2idx_path)
    
    # Load Model
    word_vectors = torch.from_numpy(np.load('data/processed/word_emb.npy')).float()
    model = BiDAF(word_vectors, char_vocab_size=100, emb_dim=300, hidden_size=100)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    total_f1 = 0
    total_em = 0
    
    print(f"🔍 Đang đánh giá trên {len(dataset.samples)} mẫu tập Dev...")
    
    for s in tqdm(dataset.samples):
        cw = torch.tensor(s['c_word']).unsqueeze(0)
        qw = torch.tensor(s['q_word']).unsqueeze(0)
        
        with torch.no_grad():
            p1, p2 = model(cw, qw)
            start = torch.argmax(p1, dim=1).item()
            end = torch.argmax(p2, dim=1).item()
        
        # Lấy text
        pred_text = " ".join(s['context_raw'][start : end+1])
        gold_text = " ".join(s['context_raw'][s['s_idx'] : s['e_idx']+1])
        
        # Tính điểm
        if pred_text == gold_text: total_em += 1
        total_f1 += calculate_f1(pred_text, gold_text)

    print(f"\n📊 KẾT QUẢ ĐÁNH GIÁ:")
    print(f" - Exact Match (EM): {total_em / len(dataset.samples) * 100:.2f}%")
    print(f" - F1 Score: {total_f1 / len(dataset.samples) * 100:.2f}%")

if __name__ == "__main__":
    evaluate('save/bidaf_epoch_13.pt', 'data/dev.json', 'data/processed/word2idx.json')