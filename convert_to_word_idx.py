import json
import os

def convert_to_word_indices(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for story in data['data']:
        for paragraph in story['paragraphs']:
            context = paragraph['context']
            # Tách context thành danh sách các từ (dựa trên khoảng trắng)
            words = context.split()
            
            # Tính toán vị trí bắt đầu của mỗi từ trong chuỗi context gốc
            word_offsets = []
            curr_pos = 0
            for w in words:
                # Tìm vị trí thực tế của từ trong context để chính xác tuyệt đối
                start_char = context.find(w, curr_pos)
                word_offsets.append((start_char, start_char + len(w)))
                curr_pos = start_char + len(w)

            for qa in paragraph['qas']:
                if not qa['answers']: continue
                
                char_start = qa['answers'][0]['answer_start']
                answer_text = qa['answers'][0]['text']
                char_end = char_start + len(answer_text)

                # Tìm word index tương ứng
                start_word_idx = -1
                end_word_idx = -1

                for i, (w_start, w_end) in enumerate(word_offsets):
                    if start_word_idx == -1 and char_start >= w_start and char_start < w_end:
                        start_word_idx = i
                    if char_end > w_start and char_end <= w_end:
                        end_word_idx = i
                
                # Lưu vào cấu trúc dữ liệu
                qa['answer_start_word'] = start_word_idx
                qa['answer_end_word'] = end_word_idx

    # Ghi đè lại file
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"✅ Đã cập nhật xong Word Index cho: {file_path}")

# Thực hiện chuyển đổi
convert_to_word_indices("data/train.json")
convert_to_word_indices("data/dev.json")