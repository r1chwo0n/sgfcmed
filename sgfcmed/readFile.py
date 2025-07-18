import os

def parse_line(line: str):
    """
    แปลงบรรทัดจากไฟล์เป็น dictionary
    """
    try:
        metadata_part, encoded_string = line.strip().split("/", 2)[1:] 
        id_str, metaphase, chrom_type, total_len, p_arm_len = metadata_part.strip().split()
        return {
            "id": int(id_str),
            "metaphase": int(metaphase),
            "chromosome_type": int(chrom_type),
            "string_length": int(total_len),
            "p_arm_length": int(p_arm_len),
            "encoded_string": encoded_string.strip()
        }
    except Exception as e:
        print(f"Error parsing line: {line}")
        raise e

def load_data_from_folder(folder_path: str):
    """
    อ่านทุกไฟล์ในโฟลเดอร์ และแปลงข้อมูลเป็น list ของ dictionary
    """
    data = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if not os.path.isfile(file_path):
            continue
        with open(file_path, "r") as f:
            for line in f:
                if line.strip():  
                    parsed = parse_line(line)
                    # parsed["source_file"] = filename 
                    data.append(parsed)
    return data

import random

def split_data(data, blind_ratio=0.2, seed=42):
    """
    แบ่งข้อมูลออกเป็น training+validation กับ blind test
    """
    random.seed(seed)
    random.shuffle(data)

    split_index = int(len(data) * (1 - blind_ratio))
    train_val_set = data[:split_index]
    blind_test_set = data[split_index:]
    
    return train_val_set, blind_test_set
