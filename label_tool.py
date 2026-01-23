import json
import os

INPUT_FILE = "vietnamese_folktale_preprocessed.json"
OUTPUT_FILE = "vietnamese_folktale_labeled.json"

def load_data():
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_data(data):
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def main():
    dataset = load_data()
    total_stories = len(dataset['data'])
    
    print(f"=== CÔNG CỤ GHI NHÃN V2.0 (Hỗ trợ nhảy truyện) ===")
    print(f"Hiện có tổng cộng {total_stories} truyện.")
    
    try:
        start_idx = int(input(f"Bạn muốn bắt đầu làm từ truyện số mấy? (1-{total_stories}): ")) - 1
    except:
        start_idx = 0

    for i in range(start_idx, total_stories):
        story = dataset['data'][i]
        # Lấy tên truyện hoặc link nếu tiêu đề bị lỗi
        title = story.get('title', 'Không rõ')
        if title == "Không rõ tiêu đề":
            title = story.get('link', '').split('/')[-1]

        for p_idx, p in enumerate(story['paragraphs']):
            # Bỏ qua nếu đã có QAs HOẶC đã được đánh dấu là skip
            if len(p['qas']) > 0 or p.get('is_skipped'):
                continue
            
            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"📖 Truyện [{i+1}/{total_stories}]: {title}")
            print(f"📄 Đoạn văn số: {p_idx + 1}")
            print("\n--- NGỮ CẢNH (CONTEXT) ---")
            print(p['context'])
            print("-" * 30)
            
            while True:
                q = input("\n👉 Nhập CÂU HỎI (s: bỏ qua đoạn này, n: nhảy sang truyện kế, exit: nghỉ): ").strip()
                
                if q.lower() == 's':
                    p['is_skipped'] = True # Đánh dấu để lần sau không hiện lại
                    break
                if q.lower() == 'n':
                    break # Nhảy sang truyện tiếp theo
                if q.lower() == 'exit':
                    save_data(dataset)
                    return

                ans = input("👉 Copy & Paste CÂU TRẢ LỜI: ").strip()
                start_idx_found = p['context'].find(ans)
                
                if start_idx_found == -1:
                    print("❌ LỖI: Câu trả lời không khớp. Hãy copy lại!")
                else:
                    p['qas'].append({
                        "id": f"q_{i}_{p_idx}_{len(p['qas'])}",
                        "question": q,
                        "answers": [{"answer_start": start_idx_found, "text": ans}]
                    })
                    print(f"✅ Đã thêm! (Vị trí: {start_idx_found})")
                    if input("Thêm câu nữa? (y/n): ").lower() != 'y': break
            
            save_data(dataset)
            if q.lower() == 'n': break # Thoát vòng lặp paragraph của truyện hiện tại

    print("\n🎉 Hoàn thành!")

if __name__ == "__main__":
    main()