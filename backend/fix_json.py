# fix_json.py として保存
import os

filename = 'optimization_tracker.py'
if os.path.exists(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    old_code = "json.dump(report, f, indent=2, ensure_ascii=False)"
    new_code = """try:
                json.dump(report, f, indent=2, ensure_ascii=False)
            except (TypeError, ValueError) as e:
                print(f"⚠️ JSON warning (non-critical): {e}")"""

    if old_code in content:
        content = content.replace(old_code, new_code)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        print("✅ JSON問題修正完了")
