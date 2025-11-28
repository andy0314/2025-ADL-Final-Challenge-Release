import json
import os
import argparse
from typing import List, Dict, Any

def merge_json_files(file1_path: str, file2_path: str, output_path: str) -> None:
    """
    載入兩個 JSON Lines (.jsonl) 檔案，將其內容合併，並寫入新的輸出檔案（JSON Lines 格式）。
    
    Args:
        file1_path (str): 第一個輸入 JSON Lines 檔案的路徑。
        file2_path (str): 第二個輸入 JSON Lines 檔案的路徑。
        output_path (str): 合併後的 JSON Lines 檔案的輸出路徑。
    """
    # 輔助函式：載入 JSON Lines (.jsonl) 格式的檔案
    def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
        records = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        # 重新拋出帶有行號的錯誤訊息，方便除錯
                        raise json.JSONDecodeError(f"Line {line_num} in {file_path}: {e.msg}", e.doc, e.pos)
        return records

    all_data = []

    print(f"--- 開始合併 JSON Lines 程序 ---")

    # 1. 處理第一個檔案
    try:
        data1 = load_jsonl(file1_path)
        all_data.extend(data1)
        print(f"成功載入 {file1_path}：包含 {len(data1)} 筆記錄。")
    except FileNotFoundError:
        print(f"錯誤：找不到檔案 {file1_path}")
        return
    except json.JSONDecodeError as e:
        print(f"錯誤：解析 {file1_path} 失敗。請確保它是有效的 JSON Lines 格式。錯誤訊息: {e}")
        return

    # 2. 處理第二個檔案
    try:
        data2 = load_jsonl(file2_path)
        all_data.extend(data2)
        print(f"成功載入 {file2_path}：包含 {len(data2)} 筆記錄。")
    except FileNotFoundError:
        print(f"錯誤：找不到檔案 {file2_path}")
        return
    except json.JSONDecodeError as e:
        print(f"錯誤：解析 {file2_path} 失敗。請確保它是有效的 JSON Lines 格式。錯誤訊息: {e}")
        return

    # 3. 寫入合併後的檔案 (JSONL 格式)
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for record in all_data:
                # 將每個記錄寫成一行 JSON 字串，並加上換行符
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f"--- 合併完成 ---")
        print(f"總共 {len(all_data)} 筆記錄已寫入 {output_path}")
    except IOError as e:
        print(f"寫入檔案 {output_path} 失敗: {e}")

def main():
    """
    解析命令行參數並呼叫檔案合併函式。
    """
    parser = argparse.ArgumentParser(description="合併兩個 JSON Lines (.jsonl) 檔案並輸出一個新檔案（JSON Lines 格式）。")
    # 參數設定不變，以符合您的執行格式
    parser.add_argument("--file1_path", type=str, required=True, help="第一個輸入 JSON Lines 檔案的路徑。")
    parser.add_argument("--file2_path", type=str, required=True, help="第二個輸入 JSON Lines 檔案的路徑。")
    parser.add_argument("--output_path", type=str, required=True, help="合併後的 JSON Lines 檔案的輸出路徑。")
    
    args = parser.parse_args()
    
    # 執行合併功能
    merge_json_files(args.file1_path, args.file2_path, args.output_path)


if __name__ == "__main__":
    main()
