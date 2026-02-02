import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import re
import json
from tqdm import tqdm
from data_loader import get_loader
from model import BiDAF

def get_latest_checkpoint(save_dir):
    if not os.path.exists(save_dir):
        return None, 0
    files = [f for f in os.listdir(save_dir) if f.endswith('.pt') and 'epoch_' in f]
    if not files:
        return None, 0
    epochs = [int(re.findall(r'\d+', f)[0]) for f in files]
    max_epoch = max(epochs)
    latest_file = os.path.join(save_dir, f'bidaf_epoch_{max_epoch}.pt')
    return latest_file, max_epoch

def train():
    # --- 1. Cấu hình ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 4
    hidden_size = 100
    step_size = 5 
    lr = 0.1  # Adadelta thường dùng lr cao
    save_dir = 'save'
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # --- 2. Tải dữ liệu và Từ điển ---
    print(" Đang tải dữ liệu và Embedding...")
    word_vectors = torch.from_numpy(np.load('data/processed/word_emb.npy')).float()
    
    # Nạp char2idx để lấy char_vocab_size chuẩn
    with open('data/processed/char2idx.json', 'r', encoding='utf-8') as f:
        char2idx = json.load(f)
    char_vocab_size = len(char2idx)

    # Khởi tạo loader với đầy đủ 3 đường dẫn file
    train_loader = get_loader(
        'data/train.json', 
        'data/processed/word2idx.json', 
        'data/processed/char2idx.json', 
        batch_size=batch_size
    )
    
    # --- 3. Khởi tạo mô hình ---
    # Chú ý: Truyền đúng tham số char_vocab_size thực tế
    model = BiDAF(
        word_vectors=word_vectors, 
        char_vocab_size=char_vocab_size, 
        hidden_size=hidden_size
    ).to(device)
    
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # --- 4. Checkpoint ---
    latest_file, start_epoch = get_latest_checkpoint(save_dir)
    target_epoch = start_epoch + step_size
    
    if latest_file:
        print(f" Nạp checkpoint: {latest_file} (Epoch {start_epoch})")
        model.load_state_dict(torch.load(latest_file, map_location=device))
        print(f" Chạy tiếp đến Epoch {target_epoch}")
    else:
        print(f" Huấn luyện từ đầu đến Epoch {target_epoch}")
        start_epoch = 0

    # --- 5. Vòng lặp huấn luyện ---
    for epoch in range(start_epoch, target_epoch):
        model.train()
        epoch_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{target_epoch}")
        for batch in progress_bar:
            optimizer.zero_grad()
            
            # Đưa tất cả dữ liệu lên Device (CPU/GPU)
            cw = batch['c_word'].to(device)
            cc = batch['c_char'].to(device) # Thêm Char Context
            qw = batch['q_word'].to(device)
            qc = batch['q_char'].to(device) # Thêm Char Question
            y1 = batch['y1'].to(device)
            y2 = batch['y2'].to(device)

            # Truyền 4 tham số vào model
            p1, p2 = model(cw, cc, qw, qc)
            
            loss = criterion(p1, y1) + criterion(p2, y2)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(train_loader)
        print(f" Epoch {epoch+1} hoàn tất. Loss: {avg_loss:.4f}")
        
        save_path = os.path.join(save_dir, f'bidaf_epoch_{epoch+1}.pt')
        torch.save(model.state_dict(), save_path)

    print(f"\n Đã hoàn thành! Tổng cộng: {target_epoch} epoch.")

if __name__ == "__main__":
    train()