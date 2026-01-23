import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Trong file data_loader.py

class FolkloreDataset(Dataset):
    # Tăng max_context_len lên 500 hoặc cao hơn tùy độ dài truyện của bạn
    def __init__(self, data_path, word2idx_path, max_context_len=600, max_question_len=50):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)['data']
        with open(word2idx_path, 'r', encoding='utf-8') as f:
            self.word2idx = json.load(f)
            
        self.max_c_len = max_context_len
        self.max_q_len = max_question_len
        
        chars = "abcdeghiklmnopqrstuvxyàáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ_ "
        self.char2idx = {char: i + 2 for i, char in enumerate(chars)}
        self.char2idx["<PAD>"] = 0
        self.char2idx["<UNK>"] = 1

        self.samples = self._process_data()
        # Thêm dòng này để kiểm tra lỗi
        print(f"✅ Đã nạp thành công {len(self.samples)} mẫu từ {data_path}")

    def _process_data(self):
        samples = []
        skipped_error = 0
        skipped_length = 0
        
        for story in self.data:
            for p in story['paragraphs']:
                # Tách từ context
                context_tokens = p['context'].lower().split()
                
                for qa in p['qas']:
                    # 1. Kiểm tra sự tồn tại của nhãn word index
                    if 'answer_start_word' not in qa or 'answer_end_word' not in qa:
                        skipped_error += 1
                        continue
                        
                    ans_start = qa['answer_start_word']
                    ans_end = qa['answer_end_word']
                    
                    # 2. Loại bỏ nhãn -1 (Lỗi không tìm thấy từ)
                    if ans_start == -1 or ans_end == -1:
                        skipped_error += 1
                        continue
                    
                    # 3. Loại bỏ nếu nhãn vượt quá độ dài tối đa (max_c_len)
                    if ans_start >= self.max_c_len or ans_end >= self.max_c_len:
                        skipped_length += 1
                        continue
                    
                    # 4. Đảm bảo start không lớn hơn end
                    if ans_start > ans_end:
                        skipped_error += 1
                        continue

                    # Xử lý câu hỏi
                    question_tokens = qa['question'].lower().split()[:self.max_q_len]
                    
                    # Chuyển thành ID
                    c_idxs = [self.word2idx.get(w, 1) for w in context_tokens[:self.max_c_len]]
                    q_idxs = [self.word2idx.get(w, 1) for w in question_tokens]
                    
                    samples.append({
                        'c_word': c_idxs,
                        'q_word': q_idxs,
                        's_idx': ans_start,
                        'e_idx': ans_end,
                        'context_raw': context_tokens[:self.max_c_len]
                    })
        
        print(f"📊 Thống kê: Giữ lại {len(samples)} mẫu. Bỏ qua {skipped_error} mẫu lỗi nhãn và {skipped_length} mẫu quá dài.")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        
        # Padding word
        c_word = np.zeros(self.max_c_len, dtype=np.int64)
        c_word[:len(s['c_word'])] = s['c_word']
        
        q_word = np.zeros(self.max_q_len, dtype=np.int64)
        q_word[:len(s['q_word'])] = s['q_word']
        
        return {
            "c_word": torch.tensor(c_word),
            "q_word": torch.tensor(q_word),
            "y1": torch.tensor(s['s_idx']),
            "y2": torch.tensor(s['e_idx'])
        }

def get_loader(data_path, word2idx_path, batch_size=4):
    dataset = FolkloreDataset(data_path, word2idx_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)