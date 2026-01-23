import torch
import torch.nn as nn
import torch.nn.functional as F

class BiDAF(nn.Module):
    def __init__(self, word_vectors, char_vocab_size, emb_dim, hidden_size):
        super(BiDAF, self).__init__()
        # 1. Word Embedding
        self.word_emb = nn.Embedding.from_pretrained(word_vectors, freeze=True)
        
        # 2. Contextual Encoding
        self.context_LSTM = nn.LSTM(emb_dim, hidden_size, batch_first=True, bidirectional=True)
        
        # 3. Attention Flow Layer (Các trọng số để tính ma trận tương đồng S)
        self.weight_sim = nn.Linear(6 * hidden_size, 1)
        
        # 4. Modeling Layer
        self.modeling_LSTM = nn.LSTM(8 * hidden_size, hidden_size, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        
        # 5. Output Layer
        self.p1_weight = nn.Linear(10 * hidden_size, 1)
        self.p2_weight = nn.Linear(10 * hidden_size, 1)

    def forward(self, c_word, q_word):
        # c_word: [batch, c_len], q_word: [batch, q_len]
        
        # --- Bước 1: Embedding Layer ---
        c_emb = self.word_emb(c_word) # [batch, c_len, 300]
        q_emb = self.word_emb(q_word) # [batch, q_len, 300]
        
        # --- Bước 2: Contextual Encoding Layer ---
        # H: [batch, c_len, 2*hidden_size], U: [batch, q_len, 2*hidden_size]
        H, _ = self.context_LSTM(c_emb)
        U, _ = self.context_LSTM(q_emb)
        
        c_len = H.size(1)
        q_len = U.size(1)
        h_dim = H.size(2)
        
        # --- Bước 3: Attention Flow Layer ---
        # Tính ma trận tương đồng S giữa Context và Query
        # Lặp lại H và U để tạo cặp tương tác (c_i, q_j)
        H_exp = H.unsqueeze(2).expand(-1, -1, q_len, -1) # [batch, c_len, q_len, 2h]
        U_exp = U.unsqueeze(1).expand(-1, c_len, -1, -1) # [batch, c_len, q_len, 2h]
        HU_sim = H_exp * U_exp                           # [batch, c_len, q_len, 2h]
        
        # S_input là sự kết hợp [h; u; h*u] theo bài báo
        S_input = torch.cat([H_exp, U_exp, HU_sim], dim=-1) # [batch, c_len, q_len, 6h]
        S = self.weight_sim(S_input).squeeze(-1)            # [batch, c_len, q_len]
        
        # Context-to-Query (C2Q) Attention
        a = F.softmax(S, dim=-1)        # [batch, c_len, q_len]
        U_tilde = torch.bmm(a, U)       # [batch, c_len, 2h]
        
        # Query-to-Context (Q2C) Attention
        b = F.softmax(torch.max(S, dim=2)[0], dim=-1) # [batch, c_len]
        h_tilde = torch.bmm(b.unsqueeze(1), H)        # [batch, 1, 2h]
        H_tilde = h_tilde.expand(-1, c_len, -1)       # [batch, c_len, 2h]
        
        # Kết hợp các lớp Attention: G = [H; U_tilde; H*U_tilde; H*H_tilde]
        G = torch.cat([H, U_tilde, H * U_tilde, H * H_tilde], dim=-1) # [batch, c_len, 8h]
        
        # --- Bước 4: Modeling Layer ---
        M, _ = self.modeling_LSTM(G) # [batch, c_len, 2h]
        
        # --- Bước 5: Output Layer ---
        # Dự đoán Start (p1) và End (p2)
        p1 = self.p1_weight(torch.cat([G, M], dim=-1)).squeeze(-1) # [batch, c_len]
        p2 = self.p2_weight(torch.cat([G, M], dim=-1)).squeeze(-1) # [batch, c_len]
        
        return p1, p2