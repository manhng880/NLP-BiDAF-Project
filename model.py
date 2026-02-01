import torch
import torch.nn as nn
import torch.nn.functional as F

class CharEmbedding(nn.Module):
    def __init__(self, char_vocab_size, char_emb_dim, out_channels, kernel_size):
        super(CharEmbedding, self).__init__()
        self.char_emb = nn.Embedding(char_vocab_size, char_emb_dim, padding_idx=0)
        # CNN để bắt đặc trưng n-gram của ký tự
        self.cnn = nn.Conv2d(1, out_channels, (char_emb_dim, kernel_size))

    def forward(self, x):
        # x shape: [batch, seq_len, word_len]
        batch_size, seq_len, word_len = x.size()
        x = x.view(-1, word_len)  # [batch * seq_len, word_len]
        x = self.char_emb(x)      # [batch * seq_len, word_len, char_emb_dim]
        
        # Biến đổi để đưa vào CNN (N, C_in, H, W)
        x = x.unsqueeze(1)        # [batch * seq_len, 1, word_len, char_emb_dim]
        x = x.transpose(2, 3)     # [batch * seq_len, 1, char_emb_dim, word_len]
        
        # CNN + ReLU
        x = F.relu(self.cnn(x)).squeeze(2) # [batch * seq_len, out_channels, W_out]
        
        # Max-pooling qua toàn bộ từ để lấy vector đại diện nhất
        x = F.max_pool1d(x, x.size(2)).squeeze(2) # [batch * seq_len, out_channels]
        
        # Trả về đúng shape cho sequence
        return x.view(batch_size, seq_len, -1) # [batch, seq_len, out_channels]

class BiDAF(nn.Module):
    def __init__(self, word_vectors, char_vocab_size, hidden_size, char_emb_dim=16, char_out_channels=100, char_kernel_size=3):
        super(BiDAF, self).__init__()
        
        # 1. Word Embedding (FastText 300d)
        self.word_emb = nn.Embedding.from_pretrained(word_vectors, freeze=True)
        
        # 2. Char Embedding Layer
        self.char_embedding = CharEmbedding(char_vocab_size, char_emb_dim, char_out_channels, char_kernel_size)
        
        # Tổng kích thước embedding = word_dim (300) + char_out_channels (100) = 400
        total_emb_dim = word_vectors.size(1) + char_out_channels
        
        # 3. Contextual Embedding
        self.context_LSTM = nn.LSTM(total_emb_dim, hidden_size, batch_first=True, bidirectional=True)
        
        # 4. Attention Flow Layer
        self.weight_sim = nn.Linear(6 * hidden_size, 1)
        
        # 5. Modeling Layer
        self.modeling_LSTM = nn.LSTM(8 * hidden_size, hidden_size, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        
        # 6. Output Layer
        self.p1_weight = nn.Linear(10 * hidden_size, 1)
        self.p2_weight = nn.Linear(10 * hidden_size, 1)

    def forward(self, c_word, c_char, q_word, q_char):
        # c_word: [batch, c_len], c_char: [batch, c_len, word_len]
        
        # --- Bước 1: Multi-stage Embedding Layer ---
        c_word_emb = self.word_emb(c_word) # [batch, c_len, 300]
        c_char_emb = self.char_embedding(c_char) # [batch, c_len, 100]
        c_emb = torch.cat([c_word_emb, c_char_emb], dim=-1) # [batch, c_len, 400]
        
        q_word_emb = self.word_emb(q_word) 
        q_char_emb = self.char_embedding(q_char)
        q_emb = torch.cat([q_word_emb, q_char_emb], dim=-1) # [batch, q_len, 400]
        
        # --- Bước 2: Contextual Embedding Layer ---
        H, _ = self.context_LSTM(c_emb) # [batch, c_len, 2h]
        U, _ = self.context_LSTM(q_emb) # [batch, q_len, 2h]
        
        c_len = H.size(1)
        q_len = U.size(1)
        
        # --- Bước 3: Attention Flow Layer ---
        H_exp = H.unsqueeze(2).expand(-1, -1, q_len, -1) 
        U_exp = U.unsqueeze(1).expand(-1, c_len, -1, -1) 
        HU_sim = H_exp * U_exp                           
        
        S_input = torch.cat([H_exp, U_exp, HU_sim], dim=-1) # [batch, c_len, q_len, 6h]
        S = self.weight_sim(S_input).squeeze(-1)            # [batch, c_len, q_len]
        
        # Context-to-Query (C2Q)
        a = F.softmax(S, dim=-1)        
        U_tilde = torch.bmm(a, U)       
        
        # Query-to-Context (Q2C)
        b = F.softmax(torch.max(S, dim=2)[0], dim=-1) 
        h_tilde = torch.bmm(b.unsqueeze(1), H)        
        H_tilde = h_tilde.expand(-1, c_len, -1)       
        
        # G = [H; U_tilde; H*U_tilde; H*H_tilde]
        G = torch.cat([H, U_tilde, H * U_tilde, H * H_tilde], dim=-1) # [batch, c_len, 8h]
        
        # --- Bước 4: Modeling Layer ---
        M, _ = self.modeling_LSTM(G) # [batch, c_len, 2h]
        
        # --- Bước 5: Output Layer ---
        p1 = self.p1_weight(torch.cat([G, M], dim=-1)).squeeze(-1) 
        p2 = self.p2_weight(torch.cat([G, M], dim=-1)).squeeze(-1) 
        
        return p1, p2