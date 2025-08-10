import json
import os

json_path = "C:/Users/yossi/Downloads/final project/FinalProject/MyData/word_list.json"

txt_path = os.path.splitext(json_path)[0] + ".txt"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

with open(txt_path, "w", encoding="utf-8") as f:
    for item in data:
        if isinstance(item, list) and len(item) >= 1:
            f.write(f"{item[0]}\n")

print(f" {txt_path}")
