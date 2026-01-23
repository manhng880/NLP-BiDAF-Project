import json
import numpy as np
from tqdm import tqdm
import os

def create_subset_embedding_v2(full_vec_path, train_file, dev_file, out_dir):
    # 1. Thu thập từ
    words = set()
    for fpath in [train_file, dev_file]:
        with open(fpath, 'r', encoding='utf-8') as f:
            data = json.load(f)['data']
            for story in data:
                for p in story['paragraphs']:
                    words.update(p['context'].lower().split())
                    for qa in p['qas']:
                        words.update(qa['question'].lower().split())

    # 2. Load toàn bộ FastText vào bộ nhớ (Cần ~4-6GB RAM)
    full_fasttext = {}
    with open(full_vec_path, 'r', encoding='utf-8', errors='ignore') as f:
        next(f)
        for line in tqdm(f, desc="Loading FastText to Memory"):
            parts = line.rstrip().split(' ')
            full_fasttext[parts[0]] = np.array([float(x) for x in parts[1:]])

    # 3. Xử lý tăng tỷ lệ phủ
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    embeddings = [np.zeros(300), np.random.uniform(-0.1, 0.1, 300)]
    found = 0

    for word in tqdm(words, desc="Mapping Words"):
        vec = None
        if word in full_fasttext:
            vec = full_fasttext[word]
        else:
            # Thử tách dấu gạch dưới
            sub_words = word.split('_')
            sub_vecs = [full_fasttext[sw] for sw in sub_words if sw in full_fasttext]
            if sub_vecs:
                vec = np.mean(sub_vecs, axis=0)
        
        if vec is not None:
            word2idx[word] = len(embeddings)
            embeddings.append(vec)
            found += 1

    # 4. Lưu
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "word_emb.npy"), np.array(embeddings))
    with open(os.path.join(out_dir, "word2idx.json"), "w", encoding='utf-8') as f:
        json.dump(word2idx, f, ensure_ascii=False)

    print(f"🚀 Tỷ lệ phủ mới: {found}/{len(words)} ({found/len(words)*100:.2f}%)")

create_subset_embedding_v2(".vector_cache/cc.vi.300.vec", "data/train.json", "data/dev.json", "data/processed")