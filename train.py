import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import re
from tqdm import tqdm
from data_loader import get_loader
from model import BiDAF

def get_latest_checkpoint(save_dir):
    """Tìm file epoch cao nhất trong thư mục save"""
    if not os.path.exists(save_dir):
        return None, 0
    
    files = [f for f in os.listdir(save_dir) if f.endswith('.pt') and 'epoch_' in f]
    if not files:
        return None, 0
    
    # Trích xuất số epoch từ tên file (vd: bidaf_epoch_10.pt -> 10)
    epochs = [int(re.findall(r'\d+', f)[0]) for f in files]
    max_epoch = max(epochs)
    latest_file = os.path.join(save_dir, f'bidaf_epoch_{max_epoch}.pt')
    
    return latest_file, max_epoch

def train():
    # --- 1. Cấu hình ---
    device = torch.device('cpu')
    batch_size = 4
    hidden_size = 100
    step_size = 10    # Số lượng epoch chạy mỗi lần
    lr = 0.5  
    save_dir = 'save'
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # --- 2. Tải dữ liệu ---
    print("🚀 Đang tải dữ liệu và Embedding...")
    word_vectors = torch.from_numpy(np.load('data/processed/word_emb.npy')).float()
    train_loader = get_loader('data/train.json', 'data/processed/word2idx.json', batch_size=batch_size)
    
    # --- 3. Khởi tạo mô hình ---
    model = BiDAF(word_vectors=word_vectors, char_vocab_size=100, emb_dim=300, hidden_size=hidden_size).to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # --- 4. Tự động tìm checkpoint mới nhất ---
    latest_file, start_epoch = get_latest_checkpoint(save_dir)
    
    # Tính toán điểm dừng cho lần chạy này
    target_epoch = start_epoch + step_size
    
    if latest_file:
        print(f"🔄 Tìm thấy checkpoint mới nhất: {latest_file}")
        print(f"📥 Nạp trọng số từ Epoch {start_epoch}...")
        model.load_state_dict(torch.load(latest_file, map_location=device))
        print(f"▶️ Chạy tiếp {step_size} epoch (từ {start_epoch + 1} đến {target_epoch})")
    else:
        print(f"🆕 Huấn luyện từ đầu. Chạy {step_size} epoch (đến Epoch {target_epoch})")
        start_epoch = 0

    # --- 5. Vòng lặp huấn luyện ---
    for epoch in range(start_epoch, target_epoch):
        model.train()
        epoch_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{target_epoch}")
        for batch in progress_bar:
            optimizer.zero_grad()
            
            cw = batch['c_word'].to(device)
            qw = batch['q_word'].to(device)
            y1 = batch['y1'].to(device)
            y2 = batch['y2'].to(device)

            p1, p2 = model(cw, qw)
            loss = criterion(p1, y1) + criterion(p2, y2)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(train_loader)
        print(f"✨ Epoch {epoch+1} hoàn tất. Loss trung bình: {avg_loss:.4f}")
        
        # Lưu file cho từng epoch
        save_path = os.path.join(save_dir, f'bidaf_epoch_{epoch+1}.pt')
        torch.save(model.state_dict(), save_path)
        print(f"💾 Đã lưu: {save_path}")

    print(f"\n✅ Đã hoàn thành đợt huấn luyện này ({step_size} epoch).")
    print(f"📍 Tổng cộng đã huấn luyện được: {target_epoch} epoch.")

if __name__ == "__main__":
    train()