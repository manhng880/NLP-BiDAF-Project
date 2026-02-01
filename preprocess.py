import json
import os
from underthesea import word_tokenize
from tqdm import tqdm

# Cấu hình file
INPUT_FILE = "vietnamese_folktale_labeled.json" 
OUTPUT_FILE = "vietnamese_folktale_char_preprocessed.json"

def clean_tokenize(text):
    # Tách từ tiếng Việt và chuyển về dạng từ đơn (tách bằng khoảng trắng)
    # format="text" của underthesea dùng dấu _ cho từ ghép, ta thay bằng khoảng trắng
    return word_tokenize(text, format="text").replace("_", " ").split()

def process_data():
    if not os.path.exists(INPUT_FILE):
        print(f" Không tìm thấy file {INPUT_FILE}!")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        full_data = json.load(f)

    processed_data = []
    count_fail = 0
    global_q_idx = 0

    print(f" Đang xử lý dữ liệu từ {INPUT_FILE}...")

    # Duyệt qua tầng 1: data
    for story in tqdm(full_data['data'], desc="Stories"):
        # Duyệt qua tầng 2: paragraphs
        for p_idx, p in enumerate(story['paragraphs']):
            context = p['context']
            
            # Nếu đoạn văn không có câu hỏi, bỏ qua
            if 'qas' not in p or not p['qas']:
                continue

            # Tokenize context một lần cho mỗi đoạn văn
            context_tokens = clean_tokenize(context)

            # Duyệt qua tầng 3: qas (câu hỏi)
            for qa in p['qas']:
                if not qa['answers']:
                    continue
                
                question = qa['question']
                answer_text = qa['answers'][0]['text']
                answer_start_char = qa['answers'][0]['answer_start']
                
                # 1. Tokenize câu hỏi
                question_tokens = clean_tokenize(question)
                
                # 2. Tokenize câu trả lời
                answer_tokens = clean_tokenize(answer_text)
                
                # 3. Tính toán vị trí Word Index của câu trả lời
                # Lấy phần văn bản trước câu trả lời và đếm số từ
                before_answer = context[:answer_start_char]
                start_word_idx = len(clean_tokenize(before_answer))
                
                # Tính từ kết thúc
                end_word_idx = start_word_idx + len(answer_tokens) - 1
                
                # 4. Kiểm tra hợp lệ (tránh lệch index do ký tự đặc biệt)
                if start_word_idx >= len(context_tokens):
                    count_fail += 1
                    continue

                # 5. Lưu vào danh sách phẳng với ID mới
                processed_item = {
                    "id": f"q_{global_q_idx}",
                    "context_tokens": context_tokens,
                    "question_tokens": question_tokens,
                    "answer_text": answer_text,
                    "answer_start_word": start_word_idx,
                    "answer_end_word": end_word_idx
                }
                processed_data.append(processed_item)
                global_q_idx += 1

    # Lưu kết quả ra file JSON phẳng
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)

    print(f"\n Hoàn thành!")
    print(f" Tổng số câu hỏi đã xử lý: {len(processed_data)}")
    print(f" File đầu ra: {OUTPUT_FILE}")
    if count_fail > 0:
        print(f" Cảnh báo: {count_fail} câu bị bỏ qua do lỗi index.")

if __name__ == "__main__":
    process_data()