import json

def sync_ids_to_tool_format(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Duyệt qua từng truyện (story_index)
    for s_idx, story in enumerate(data['data']):
        # Duyệt qua từng đoạn văn (p_idx)
        for p_idx, p in enumerate(story['paragraphs']):
            # Duyệt qua từng câu hỏi (q_idx)
            if 'qas' in p:
                for q_idx, qa in enumerate(p['qas']):
                    # Tạo ID chuẩn: q_{story_index}_{paragraph_index}_{question_index}
                    # Lưu ý: s_idx, p_idx, q_idx bắt đầu từ 0
                    qa['id'] = f"q_{s_idx}_{p_idx}_{q_idx}"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    print(f"✅ Đã đồng bộ ID về định dạng tool mới cho {len(data['data'])} truyện.")

# Chạy trên file 50 truyện đã tokenized của bạn
sync_ids_to_tool_format("vietnamese_folktale_50_stories_final.json", "vietnamese_folktale_50_final_v2.json")