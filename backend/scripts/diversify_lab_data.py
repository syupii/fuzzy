#!/usr/bin/env python3
"""
研究室データ多様化スクリプト（簡易版）

使い方:
  python diversify_simple.py [入力ファイル]

例:
  python diversify_simple.py data/labs_database.json
  python diversify_simple.py  # デフォルト: data/labs_database.json
"""

import json
import sys
import random
from pathlib import Path
from datetime import datetime

def diversify_value(base_value, variation=2.0):
    """値に多様性を持たせる（境界値を優先）"""
    # 40%の確率で境界値（3-5, 7-9）を生成
    if random.random() < 0.4:
        boundaries = [4.0, 8.0]  # 境界値の中心
        boundary = random.choice(boundaries)
        return round(boundary + random.uniform(-0.8, 0.8), 1)
    else:
        # 通常の変動
        return round(max(1.0, min(10.0, base_value + random.uniform(-variation, variation))), 1)

def main():
    print("🔧 研究室データ多様化スクリプト（簡易版）")
    print("=" * 60)
    
    # 入力ファイルの決定
    if len(sys.argv) > 1:
        input_path = Path(sys.argv[1])
    else:
        # デフォルトパス（複数候補を試す）
        candidates = [
            Path("data/labs_database.json"),
            Path("../data/labs_database.json"),
            Path("backend/data/labs_database.json"),
        ]
        input_path = None
        for candidate in candidates:
            if candidate.exists():
                input_path = candidate
                break
        
        if input_path is None:
            print("❌ エラー: labs_database.json が見つかりません")
            print("\n候補:")
            for candidate in candidates:
                print(f"  - {candidate.absolute()}")
            print("\n使い方: python diversify_simple.py <ファイルパス>")
            sys.exit(1)
    
    # ファイルの存在確認
    if not input_path.exists():
        print(f"❌ エラー: ファイルが見つかりません")
        print(f"   パス: {input_path.absolute()}")
        sys.exit(1)
    
    print(f"✅ 入力ファイル: {input_path}")
    
    # 出力ファイル
    output_path = input_path.parent / "labs_database_diversified.json"
    
    # データ読み込み
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    labs = data.get('labs', [])
    print(f"📊 研究室数: {len(labs)}件")
    
    # 多様化
    criteria = [
        "research_intensity", "advisor_style", "team_work",
        "workload", "theory_practice", "skill_development",
        "lab_atmosphere", "flexibility", "publication_opportunity",
        "interdisciplinary", "communication_style", "research_field_match"
    ]
    
    boundary_count = 0
    
    for lab in labs:
        if 'features' not in lab:
            print(f"⚠️ {lab.get('name', '不明')}: features がありません。スキップ")
            continue
        
        features = lab['features']
        has_boundary = False
        
        for criterion in criteria:
            if criterion in features:
                original = features[criterion]
                # 元の値を基準に多様化
                features[criterion] = diversify_value(original)
                
                # 境界値判定
                new_val = features[criterion]
                if (3.0 <= new_val <= 5.0) or (7.0 <= new_val <= 9.0):
                    has_boundary = True
        
        if has_boundary:
            boundary_count += 1
    
    # 統計
    print(f"🎯 境界値を含む研究室: {boundary_count}/{len(labs)}件 ({boundary_count/len(labs)*100:.1f}%)")
    
    # 保存
    data['last_updated'] = datetime.now().isoformat()
    data['diversification_info'] = {
        'method': 'simple',
        'diversified_at': datetime.now().isoformat(),
        'boundary_labs': boundary_count
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 出力: {output_path}")
    print("\n" + "=" * 60)
    print("✅ 完了！")
    print("\n次のステップ:")
    print(f"1. バックアップ: cp {input_path.name} {input_path.stem}_backup.json")
    print(f"2. 適用: cp {output_path.name} {input_path.name}")
    print(f"3. サーバー再起動")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)