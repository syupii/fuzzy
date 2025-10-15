#!/usr/bin/env python3
"""
大量研究室データ生成スクリプト

目的: 100件以上の多様な研究室データを生成し、
      ファジィ決定木のパス数を最大化する
"""

import json
import random
from pathlib import Path
from datetime import datetime

# 12項目の評価基準
CRITERIA = [
    "research_intensity",
    "advisor_style",
    "team_work",
    "workload",
    "theory_practice",
    "skill_development",
    "lab_atmosphere",
    "flexibility",
    "publication_opportunity",
    "interdisciplinary",
    "communication_style",
    "research_field_match"
]

# 研究分野とfield_id
RESEARCH_FIELDS = {
    "ai_machine_learning": "人工知能・機械学習",
    "image_processing": "画像・映像処理",
    "network_security": "ネットワーク・セキュリティ",
    "database_info_system": "データベース・情報システム",
    "embedded_iot": "組込み・IoT",
    "education_linguistics": "教育・言語学",
    "natural_science_math": "自然科学・数理",
    "tourism_regional": "観光情報・地域システム",
    "management_decision": "経営情報・意思決定支援",
    "audio_processing": "音声・音響情報処理",
    "system_operation": "システム運用・情報倫理",
    "web_design_ui_ux": "Webデザイン・UI/UX",
    "design_visual": "デザイン・視覚表現",
    "video_animation": "映像・アニメーション",
    "computer_music": "コンピュータ音楽・サウンドアート",
    "game_dev_esports": "ゲーム開発・eスポーツ",
    "vr_ar_media": "VR/AR・メディアアート",
    "philosophy_humanities": "哲学・人文・環境行動学",
    "sports_science": "スポーツ・体育科学",
}

# 教授名のバリエーション
PROFESSOR_NAMES = [
    "田中", "佐藤", "鈴木", "高橋", "渡辺", "伊藤", "山本", "中村", "小林", "加藤",
    "吉田", "山田", "佐々木", "山口", "松本", "井上", "木村", "林", "清水", "山崎",
    "森", "池田", "橋本", "阿部", "石川", "山下", "中島", "石井", "藤田", "小川",
    "前田", "岡田", "長谷川", "村上", "近藤", "石田", "後藤", "坂本", "遠藤", "青木",
    "藤井", "西村", "福田", "太田", "三浦", "岡本", "原田", "竹内", "中野", "金子"
]

# 研究室の特性パターン（30種類以上）
LAB_PATTERNS = {
    # === 基本パターン（8種類） ===
    "ultra_research_intensive": {
        "description": "超研究集約型",
        "research_intensity": (9.0, 10.0),
        "publication_opportunity": (9.0, 10.0),
        "workload": (8.5, 10.0),
        "flexibility": (1.0, 3.0),
        "advisor_style": (3.0, 5.0),
    },
    
    "research_intensive": {
        "description": "研究集約型",
        "research_intensity": (7.5, 9.0),
        "publication_opportunity": (7.5, 9.5),
        "workload": (7.0, 9.0),
        "flexibility": (3.0, 5.5),
    },
    
    "balanced": {
        "description": "バランス型",
        "research_intensity": (5.0, 7.0),
        "advisor_style": (5.0, 7.0),
        "team_work": (5.0, 7.0),
        "workload": (5.0, 7.0),
    },
    
    "super_flexible": {
        "description": "超柔軟型",
        "flexibility": (9.0, 10.0),
        "advisor_style": (8.5, 10.0),
        "workload": (1.0, 3.0),
        "research_intensity": (3.0, 5.0),
    },
    
    "flexible": {
        "description": "柔軟型",
        "flexibility": (7.5, 9.5),
        "advisor_style": (7.0, 9.0),
        "workload": (3.0, 5.5),
    },
    
    "collaborative": {
        "description": "協働型",
        "team_work": (8.0, 10.0),
        "communication_style": (8.0, 10.0),
        "lab_atmosphere": (7.5, 9.5),
        "interdisciplinary": (7.0, 9.0),
    },
    
    "theory_focused": {
        "description": "理論重視型",
        "theory_practice": (1.0, 3.5),
        "research_intensity": (7.5, 9.5),
        "publication_opportunity": (7.0, 9.0),
    },
    
    "practice_focused": {
        "description": "実践重視型",
        "theory_practice": (7.5, 10.0),
        "skill_development": (8.0, 10.0),
        "team_work": (7.0, 9.0),
    },
    
    # === 境界値パターン（6種類） ===
    "boundary_low_mid_1": {
        "description": "境界値型1（全項目3-5点）",
        "research_intensity": (3.0, 5.0),
        "advisor_style": (3.0, 5.0),
        "team_work": (3.0, 5.0),
        "workload": (3.0, 5.0),
        "theory_practice": (3.0, 5.0),
        "publication_opportunity": (3.0, 5.0),
    },
    
    "boundary_low_mid_2": {
        "description": "境界値型2（一部3-5点）",
        "research_intensity": (3.5, 4.5),
        "advisor_style": (3.5, 4.5),
        "team_work": (3.5, 4.5),
    },
    
    "boundary_mid_high_1": {
        "description": "境界値型3（全項目7-9点）",
        "research_intensity": (7.0, 9.0),
        "advisor_style": (7.0, 9.0),
        "team_work": (7.0, 9.0),
        "workload": (7.0, 9.0),
        "theory_practice": (7.0, 9.0),
        "publication_opportunity": (7.0, 9.0),
    },
    
    "boundary_mid_high_2": {
        "description": "境界値型4（一部7-9点）",
        "research_intensity": (7.5, 8.5),
        "publication_opportunity": (7.5, 8.5),
        "workload": (7.5, 8.5),
    },
    
    "boundary_mix_1": {
        "description": "境界値混合型1",
        "research_intensity": (3.5, 4.5),
        "advisor_style": (7.5, 8.5),
        "team_work": (5.5, 6.5),
        "workload": (3.5, 4.5),
    },
    
    "boundary_mix_2": {
        "description": "境界値混合型2",
        "research_intensity": (7.5, 8.5),
        "advisor_style": (3.5, 4.5),
        "team_work": (5.5, 6.5),
        "publication_opportunity": (7.5, 8.5),
    },
    
    # === 特殊パターン（10種類） ===
    "skill_development_focus": {
        "description": "スキル開発重視型",
        "skill_development": (8.5, 10.0),
        "theory_practice": (7.0, 9.0),
        "team_work": (7.0, 9.0),
    },
    
    "publication_powerhouse": {
        "description": "論文強豪型",
        "publication_opportunity": (9.0, 10.0),
        "research_intensity": (8.5, 10.0),
        "workload": (8.0, 10.0),
        "flexibility": (2.0, 4.0),
    },
    
    "interdisciplinary_focus": {
        "description": "学際重視型",
        "interdisciplinary": (8.5, 10.0),
        "communication_style": (7.5, 9.5),
        "team_work": (7.0, 9.0),
    },
    
    "startup_style": {
        "description": "スタートアップ型",
        "flexibility": (8.0, 10.0),
        "theory_practice": (8.0, 10.0),
        "lab_atmosphere": (8.5, 10.0),
        "team_work": (8.0, 10.0),
    },
    
    "traditional_academic": {
        "description": "伝統的アカデミック型",
        "theory_practice": (1.0, 3.0),
        "advisor_style": (3.0, 5.0),
        "publication_opportunity": (8.0, 10.0),
        "research_intensity": (8.0, 10.0),
    },
    
    "industry_collaboration": {
        "description": "産学連携型",
        "theory_practice": (7.5, 9.5),
        "interdisciplinary": (7.5, 9.5),
        "skill_development": (8.0, 10.0),
        "team_work": (7.5, 9.5),
    },
    
    "beginner_friendly": {
        "description": "初心者歓迎型",
        "workload": (2.0, 4.0),
        "advisor_style": (7.5, 9.5),
        "flexibility": (7.5, 9.5),
        "research_intensity": (3.0, 5.0),
    },
    
    "competitive": {
        "description": "競争激励型",
        "workload": (8.5, 10.0),
        "research_intensity": (9.0, 10.0),
        "publication_opportunity": (8.5, 10.0),
        "communication_style": (3.0, 5.0),
    },
    
    "creative_focus": {
        "description": "創造性重視型",
        "flexibility": (8.0, 10.0),
        "lab_atmosphere": (8.5, 10.0),
        "advisor_style": (7.5, 9.5),
        "interdisciplinary": (7.5, 9.5),
    },
    
    "systematic": {
        "description": "体系的型",
        "advisor_style": (3.0, 5.0),
        "team_work": (7.0, 9.0),
        "communication_style": (7.0, 9.0),
        "workload": (6.0, 8.0),
    },
    
    # === 極端パターン（6種類） ===
    "all_high": {
        "description": "全高値型",
        "research_intensity": (8.5, 10.0),
        "advisor_style": (8.5, 10.0),
        "team_work": (8.5, 10.0),
        "workload": (8.5, 10.0),
    },
    
    "all_low": {
        "description": "全低値型",
        "research_intensity": (1.0, 2.5),
        "advisor_style": (1.0, 2.5),
        "team_work": (1.0, 2.5),
        "workload": (1.0, 2.5),
    },
    
    "extreme_contrast_1": {
        "description": "極端対比型1",
        "research_intensity": (9.0, 10.0),
        "workload": (9.0, 10.0),
        "flexibility": (1.0, 2.0),
        "advisor_style": (1.0, 2.0),
    },
    
    "extreme_contrast_2": {
        "description": "極端対比型2",
        "flexibility": (9.0, 10.0),
        "advisor_style": (9.0, 10.0),
        "research_intensity": (1.0, 2.0),
        "workload": (1.0, 2.0),
    },
    
    "bimodal_1": {
        "description": "二峰型1（高低混在）",
        "research_intensity": (9.0, 10.0),
        "advisor_style": (1.0, 2.0),
        "team_work": (9.0, 10.0),
        "workload": (1.0, 2.0),
    },
    
    "bimodal_2": {
        "description": "二峰型2（中央避け）",
        "research_intensity": (1.0, 3.0),
        "advisor_style": (8.0, 10.0),
        "team_work": (2.0, 3.0),
        "workload": (8.0, 10.0),
    },
}


def generate_value(pattern_range=None, boundary_bias=0.5):
    """値を生成（境界値バイアス付き）"""
    if pattern_range:
        min_val, max_val = pattern_range
        
        # 境界値を優先的に生成
        if random.random() < boundary_bias:
            # 境界値の候補
            if min_val <= 4.0 and max_val >= 4.0:
                return round(random.uniform(3.8, 4.2), 1)
            elif min_val <= 8.0 and max_val >= 8.0:
                return round(random.uniform(7.8, 8.2), 1)
            elif min_val <= 5.5 and max_val >= 5.5:
                return round(random.uniform(5.3, 5.7), 1)
        
        return round(random.uniform(min_val, max_val), 1)
    else:
        return round(random.uniform(3.0, 10.0), 1)


def generate_lab(lab_id, pattern_name, field_id, professor_name, index):
    """研究室データを生成"""
    
    pattern = LAB_PATTERNS[pattern_name]
    field_name = RESEARCH_FIELDS[field_id]
    
    # features生成
    features = {}
    for criterion in CRITERIA:
        if criterion in pattern:
            features[criterion] = generate_value(pattern[criterion])
        else:
            # パターンに含まれない項目は中程度の範囲
            features[criterion] = generate_value((4.0, 7.0), boundary_bias=0.3)
    
    # 研究室データ
    lab = {
        "id": lab_id,
        "name": f"{field_name}研究室{index}",
        "professor": f"{professor_name}教授",
        "research_area": field_name,
        "specialization": f"{field_name}専門",
        "research_fields": [field_name],
        "field_id": field_id,
        "description": f"{field_name}の研究を行う{pattern['description']}研究室",
        "features": features,
        "metadata": {
            "faculty_count": 1,
            "student_count": random.randint(5, 15),
            "recent_publications": random.randint(5, 20),
            "funding_level": random.choice(["高", "中", "低"]),
            "equipment_rating": random.randint(5, 10),
            "pattern_type": pattern_name,
            "pattern_description": pattern["description"]
        }
    }
    
    return lab


def generate_many_labs(num_labs=100):
    """大量の研究室データを生成"""
    
    print(f"📊 {num_labs}件の研究室データを生成中...")
    print("=" * 60)
    
    labs = []
    pattern_names = list(LAB_PATTERNS.keys())
    field_ids = list(RESEARCH_FIELDS.keys())
    
    pattern_count = {p: 0 for p in pattern_names}
    field_count = {f: 0 for f in field_ids}
    
    for i in range(num_labs):
        # パターンをローテーション＋ランダム
        if i < len(pattern_names):
            pattern_name = pattern_names[i]
        else:
            # 境界値パターンを優先
            boundary_patterns = [p for p in pattern_names if "boundary" in p]
            if random.random() < 0.4:
                pattern_name = random.choice(boundary_patterns)
            else:
                pattern_name = random.choice(pattern_names)
        
        # 分野をローテーション＋ランダム
        field_id = field_ids[i % len(field_ids)]
        
        # 教授名をローテーション
        professor_name = PROFESSOR_NAMES[i % len(PROFESSOR_NAMES)]
        
        # 研究室生成
        lab_id = f"lab_{i+1:03d}"
        lab = generate_lab(lab_id, pattern_name, field_id, professor_name, i+1)
        
        labs.append(lab)
        pattern_count[pattern_name] += 1
        field_count[field_id] += 1
    
    # 統計
    print(f"✅ {len(labs)}件の研究室データを生成")
    print(f"\n📈 パターン別分布（上位10件）:")
    sorted_patterns = sorted(pattern_count.items(), key=lambda x: x[1], reverse=True)
    for pattern, count in sorted_patterns[:10]:
        print(f"  {pattern}: {count}件")
    
    print(f"\n🏫 分野別分布:")
    sorted_fields = sorted(field_count.items(), key=lambda x: x[1], reverse=True)
    for field, count in sorted_fields[:10]:
        print(f"  {RESEARCH_FIELDS[field]}: {count}件")
    
    # 境界値統計
    boundary_labs = sum(1 for lab in labs if "boundary" in lab["metadata"]["pattern_type"])
    print(f"\n🎯 境界値パターン: {boundary_labs}/{len(labs)}件 ({boundary_labs/len(labs)*100:.1f}%)")
    
    return labs, pattern_count, field_count


def save_labs(labs, output_file):
    """データを保存"""
    
    data = {
        "version": "2.0.0",
        "last_updated": datetime.now().isoformat(),
        "description": "研究室選択支援システム用研究室データベース（大量生成版）",
        "total_labs": len(labs),
        "evaluation_criteria": {
            "basic": CRITERIA[:5],
            "extended": CRITERIA[5:10],
            "special": CRITERIA[10:]
        },
        "research_fields": list(RESEARCH_FIELDS.values()),
        "generation_info": {
            "generated_at": datetime.now().isoformat(),
            "num_patterns": len(LAB_PATTERNS),
            "num_fields": len(RESEARCH_FIELDS)
        },
        "labs": labs
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 保存完了: {output_file}")


def main():
    import sys
    
    print("🏭 大量研究室データ生成スクリプト")
    print("=" * 60)
    
    # 研究室数を指定（デフォルト100）
    if len(sys.argv) > 1:
        try:
            num_labs = int(sys.argv[1])
        except:
            num_labs = 100
    else:
        num_labs = 100
    
    print(f"生成数: {num_labs}件\n")
    
    # パスの設定
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    output_file = project_root / "data" / "labs_database.json"
    
    # バックアップ
    if output_file.exists():
        backup_file = project_root / "data" / f"labs_database_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        import shutil
        shutil.copy(output_file, backup_file)
        print(f"📦 バックアップ作成: {backup_file.name}\n")
    
    # 生成
    labs, pattern_count, field_count = generate_many_labs(num_labs)
    
    # 保存
    save_labs(labs, output_file)
    
    print("\n" + "=" * 60)
    print("✅ 完了！")
    print(f"\n次のステップ:")
    print(f"1. サーバーを再起動（自動リロードされます）")
    print(f"2. 評価を実行してパス数を確認")
    print(f"\n期待されるパス数分布:")
    print(f"  - 1-2パス: ~20%")
    print(f"  - 3-5パス: ~50%")
    print(f"  - 6-8パス: ~25%")
    print(f"  - 9+パス: ~5%")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ エラー: {e}")
        import traceback
        traceback.print_exc()