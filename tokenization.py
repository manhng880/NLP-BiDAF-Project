import json
import os
from underthesea import word_tokenize

# Cấu hình file
INPUT_FILE = "vietnamese_folktale_labeled.json" # File gốc bạn đang làm
OUTPUT_FILE = "vietnamese_folktale_50_stories_final.json"

def align_and_tokenize(text, original_start, answer_text):
    # 1. Tách từ cho toàn bộ context
    # format="text" sẽ dùng dấu gạch dưới cho từ ghép và tách dấu câu
    tokenized_text = word_tokenize(text, format="text")
    
    # 2. Tìm vị trí mới của câu trả lời
    # Vì bạn đã label rất chuẩn, chúng ta sẽ tìm bản 'đã tách từ' của câu trả lời trong context mới
    tokenized_answer = word_tokenize(answer_text, format="text")
    
    new_start = tokenized_text.find(tokenized_answer)
    
    # Nếu không tìm thấy trực tiếp (do underthesea tách khác nhau), ta dùng thuật toán bù trừ
    if new_start == -1:
        # Thử tìm bản không dấu gạch dưới
        clean_ans = tokenized_answer.replace("_", " ")
        clean_context = tokenized_text.replace("_", " ")
        new_start = clean_context.find(clean_ans)
        
    return tokenized_text, new_start, tokenized_answer

def process_stage_1():
    if not os.path.exists(INPUT_FILE):
        print("❌ Không tìm thấy file gốc!")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Chỉ lấy 50 truyện đầu tiên
    subset_data = data['data'][:50]
    new_data = {"version": "v1.0-50stories", "data": []}
    
    count_success = 0
    count_fail = 0

    for story in subset_data:
        new_story = {"title": story.get('title', 'Không rõ tiêu đề'), "paragraphs": []}
        
        for p in story['paragraphs']:
            if not p['qas']: continue # Bỏ qua đoạn không có câu hỏi
            
            # Tách từ context
            tokenized_context = word_tokenize(p['context'], format="text")
            new_p = {"context": tokenized_context, "qas": []}
            
            for qa in p['qas']:
                new_qa = {"id": qa['id'], "question": word_tokenize(qa['question'], format="text"), "answers": []}
                
                for ans in qa['answers']:
                    # Căn chỉnh lại index
                    _, new_start, new_ans_text = align_and_tokenize(p['context'], ans['answer_start'], ans['text'])
                    
                    if new_start != -1:
                        new_qa['answers'].append({
                            "answer_start": new_start,
                            "text": new_ans_text
                        })
                        count_success += 1
                    else:
                        count_fail += 1
                
                if new_qa['answers']:
                    new_p['qas'].append(new_qa)
            
            new_story['paragraphs'].append(new_p)
        new_data['data'].append(new_story)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)

    print(f"\n✅ Hoàn thành 50 truyện!")
    print(f"📊 Thành công: {count_success} câu hỏi")
    print(f"⚠️ Thất bại (lệch index): {count_fail} câu")

if __name__ == "__main__":
    process_stage_1()