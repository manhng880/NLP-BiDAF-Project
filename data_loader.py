import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class FolkloreDataset(Dataset):
    def __init__(self, data_path, word2idx_path, char2idx_path, max_context_len=600, max_question_len=50, max_word_len=16):
        # 1. Load dữ liệu phẳng [{}, {}, ...] từ preprocess/split
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # 2. Load các bộ từ điển đã tạo từ setup_data
        with open(word2idx_path, 'r', encoding='utf-8') as f:
            self.word2idx = json.load(f)
        with open(char2idx_path, 'r', encoding='utf-8') as f:
            self.char2idx = json.load(f)
            
        self.max_c_len = max_context_len
        self.max_q_len = max_question_len
        self.max_w_len = max_word_len 

        # Tiền xử lý dữ liệu để khi train/eval chạy nhanh hơn
        self.samples = self._process_data()
        print(f" Đã nạp thành công {len(self.samples)} mẫu từ {data_path}")

    def _process_data(self):
        samples = []
        skipped_error = 0
        skipped_length = 0
        
        for item in self.data:
            # Lấy token đã được tách từ preprocess.py
            context_tokens = item['context_tokens']
            question_tokens = item['question_tokens']
            
            # Kiểm tra sự tồn tại của nhãn (rất quan trọng)
            if 'answer_start_word' not in item or 'answer_end_word' not in item:
                skipped_error += 1
                continue
                
            ans_start = item['answer_start_word']
            ans_end = item['answer_end_word']
            
            # Lọc các mẫu lỗi (index âm hoặc start > end)
            if ans_start == -1 or ans_end == -1 or ans_start > ans_end:
                skipped_error += 1
                continue
            
            # Lọc các mẫu quá dài so với giới hạn của mô hình (Tránh lỗi văng index)
            if ans_start >= self.max_c_len or ans_end >= self.max_c_len:
                skipped_length += 1
                continue

            # 1. Chuyển Word sang ID
            c_word_idxs = [self.word2idx.get(w, 1) for w in context_tokens[:self.max_c_len]]
            q_word_idxs = [self.word2idx.get(w, 1) for w in question_tokens[:self.max_q_len]]
            
            # 2. Chuyển Ký tự sang ID (Char-level)
            c_char_idxs = []
            for w in context_tokens[:self.max_c_len]:
                # Lấy ID ký tự, padding/truncate từng từ về độ dài max_word_len
                w_chars = [self.char2idx.get(c, 1) for c in list(w)[:self.max_w_len]]
                w_chars += [0] * (self.max_w_len - len(w_chars))
                c_char_idxs.append(w_chars)
            
            q_char_idxs = []
            for w in question_tokens[:self.max_q_len]:
                w_chars = [self.char2idx.get(c, 1) for c in list(w)[:self.max_w_len]]
                w_chars += [0] * (self.max_w_len - len(w_chars))
                q_char_idxs.append(w_chars)

            samples.append({
                'c_word': c_word_idxs,
                'c_char': c_char_idxs,
                'q_word': q_word_idxs,
                'q_char': q_char_idxs,
                'y1': ans_start,
                'y2': ans_end
            })
        
        print(f" Thống kê: Giữ lại {len(samples)} mẫu. Bỏ qua {skipped_error} lỗi, {skipped_length} quá dài.")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        
        # --- Padding Context (Đưa về tensor cố định) ---
        c_word = np.zeros(self.max_c_len, dtype=np.int64)
        c_word[:len(s['c_word'])] = s['c_word']
        
        c_char = np.zeros((self.max_c_len, self.max_w_len), dtype=np.int64)
        c_char[:len(s['c_char']), :] = s['c_char']
        
        # --- Padding Question (Đưa về tensor cố định) ---
        q_word = np.zeros(self.max_q_len, dtype=np.int64)
        q_word[:len(s['q_word'])] = s['q_word']
        
        q_char = np.zeros((self.max_q_len, self.max_w_len), dtype=np.int64)
        q_char[:len(s['q_char']), :] = s['q_char']
        
        return {
            "c_word": torch.tensor(c_word),
            "c_char": torch.tensor(c_char),
            "q_word": torch.tensor(q_word),
            "q_char": torch.tensor(q_char),
            "y1": torch.tensor(s['y1']),
            "y2": torch.tensor(s['y2'])
        }

def get_loader(data_path, word2idx_path, char2idx_path, batch_size=4, shuffle=True):
    dataset = FolkloreDataset(data_path, word2idx_path, char2idx_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)