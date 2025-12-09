#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
遺伝的アルゴリズム結果の可視化
適合度推移、最適プロファイル、比較分析を可視化

使用方法:
    python visualize_ga_results.py --lab_id lab_001
    python visualize_ga_results.py --compare lab_001 lab_002 lab_003
    python visualize_ga_results.py --all --output results/visualizations/
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import argparse
from typing import Dict, List, Any
import seaborn as sns

# 日本語フォント設定
mpl.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 研究分野定義
RESEARCH_FIELDS = {
    "ai_ml": "AI・機械学習",
    "image_processing": "画像処理",
    "network_security": "ネットワークセキュリティ",
    "database_systems": "データベース",
    "embedded_iot": "組込み・IoT",
    "education_linguistics": "教育・言語学",
    "natural_science_math": "自然科学・数理",
    "tourism_regional": "観光情報",
    "business_decision": "経営情報",
    "audio_processing": "音声処理",
    "system_ethics": "情報倫理",
    "medical_healthcare": "医療情報",
    "web_design": "Webデザイン",
    "design_visual": "視覚デザイン",
    "video_animation": "映像制作",
    "computer_music": "音楽制作",
    "game_esports": "ゲーム開発",
    "vr_ar_media": "VR/AR",
    "philosophy_humanities": "哲学・人文",
    "sports_science": "スポーツ科学"
}

CRITERIA_NAMES = {
    "research_intensity": "研究強度",
    "advisor_style": "指導スタイル",
    "team_work": "チームワーク",
    "workload": "ワークロード",
    "theory_practice": "理論・実践",
    "research_field_match": "分野重視度",
    "skill_development": "スキル開発",
    "lab_atmosphere": "雰囲気",
    "flexibility": "柔軟性",
    "publication_opportunity": "論文機会",
    "interdisciplinary": "学際性",
    "communication_style": "コミュニケーション"
}


def load_result(lab_id: str, base_dir: str = "results/genetic_optimization") -> Dict[str, Any]:
    """結果ファイルを読み込み"""
    filepath = Path(base_dir) / lab_id / f"{lab_id}_optimal_student.json"
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_fitness_evolution(result: Dict[str, Any], output_path: Path):
    """適合度の進化を可視化"""
    fitness_history = result["fitness_history"]
    lab_name = result["lab_name"]
    
    plt.figure(figsize=(12, 6))
    
    generations = list(range(len(fitness_history)))
    plt.plot(generations, fitness_history, 'b-', linewidth=2, label='Best Fitness')
    
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Fitness Score', fontsize=12)
    plt.title(f'Genetic Algorithm Fitness Evolution\n{lab_name} ({result["lab_id"]})', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # 統計情報を追加
    final_fitness = fitness_history[-1]
    convergence_gen = next((i for i, f in enumerate(fitness_history) if f >= final_fitness * 0.99), len(fitness_history))
    
    textstr = f'Final Fitness: {final_fitness:.4f}\nConvergence at Gen: {convergence_gen}'
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path / f"{result['lab_id']}_fitness_evolution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 適合度進化グラフを保存: {output_path / f'{result['lab_id']}_fitness_evolution.png'}")


def plot_student_profile_radar(result: Dict[str, Any], output_path: Path):
    """最適学生プロファイルをレーダーチャートで可視化"""
    profile = result["optimal_student_profile"]
    lab_name = result["lab_name"]
    
    # 12項目を取得
    criteria = list(CRITERIA_NAMES.keys())
    values = [profile.get(c, 5.5) for c in criteria]
    
    # レーダーチャート用に角度を計算
    angles = np.linspace(0, 2 * np.pi, len(criteria), endpoint=False).tolist()
    values += values[:1]  # 閉じるために最初の値を追加
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    ax.plot(angles, values, 'o-', linewidth=2, color='#2E86AB', label='Optimal Student')
    ax.fill(angles, values, alpha=0.25, color='#2E86AB')
    
    # ラベル設定
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([CRITERIA_NAMES[c] for c in criteria], fontsize=10)
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.title(f'Optimal Student Profile\n{lab_name} ({result["lab_id"]})', 
              fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path / f"{result['lab_id']}_profile_radar.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ プロファイルレーダーチャートを保存: {output_path / f'{result['lab_id']}_profile_radar.png'}")


def plot_field_interests(result: Dict[str, Any], output_path: Path):
    """分野興味を棒グラフで可視化"""
    profile = result["optimal_student_profile"]
    field_interests = profile.get("field_interests", {})
    lab_name = result["lab_name"]
    
    if not field_interests:
        print("  分野興味データがありません")
        return
    
    # データ準備
    fields = list(field_interests.keys())
    interests = list(field_interests.values())
    field_labels = [RESEARCH_FIELDS.get(f, f) for f in fields]
    
    # 研究室の分野をハイライト
    lab_fields = set(result.get("research_fields", []))
    colors = ['#E63946' if f in lab_fields else '#2E86AB' for f in fields]
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(field_labels, interests, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    plt.xlabel('Interest Level (1-10)', fontsize=12)
    plt.ylabel('Research Field', fontsize=12)
    plt.title(f'Optimal Student Field Interests\n{lab_name} ({result["lab_id"]})', 
              fontsize=14, fontweight='bold')
    plt.xlim(0, 10)
    plt.grid(True, axis='x', alpha=0.3)
    
    # 凡例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#E63946', alpha=0.7, label="Lab's Field"),
        Patch(facecolor='#2E86AB', alpha=0.7, label='Related Field')
    ]
    plt.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path / f"{result['lab_id']}_field_interests.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 分野興味グラフを保存: {output_path / f'{result['lab_id']}_field_interests.png'}")


def plot_criteria_comparison(results: List[Dict[str, Any]], output_path: Path):
    """複数研究室の最適学生プロファイルを比較"""
    if len(results) < 2:
        print("比較には2つ以上の研究室が必要です")
        return
    
    criteria = list(CRITERIA_NAMES.keys())
    x = np.arange(len(criteria))
    width = 0.8 / len(results)
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(results)))
    
    for i, result in enumerate(results):
        profile = result["optimal_student_profile"]
        values = [profile.get(c, 5.5) for c in criteria]
        
        offset = (i - len(results)/2 + 0.5) * width
        ax.bar(x + offset, values, width, label=result["lab_name"], 
               color=colors[i], alpha=0.8, edgecolor='black', linewidth=1)
    
    ax.set_xlabel('Criteria', fontsize=12)
    ax.set_ylabel('Value (1-10)', fontsize=12)
    ax.set_title('Optimal Student Profile Comparison Across Labs', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([CRITERIA_NAMES[c] for c in criteria], rotation=45, ha='right', fontsize=10)
    ax.set_ylim(0, 10)
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(fontsize=9, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_path / "labs_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 研究室比較グラフを保存: {output_path / 'labs_comparison.png'}")


def generate_summary_report(result: Dict[str, Any], output_path: Path):
    """サマリーレポートをテキストで生成"""
    lab_id = result["lab_id"]
    lab_name = result["lab_name"]
    profile = result["optimal_student_profile"]
    config = result["optimization_config"]
    
    report = f"""
{'='*70}
遺伝的アルゴリズム分析レポート
{'='*70}

【研究室情報】
  研究室ID: {lab_id}
  研究室名: {lab_name}
  専門分野: {', '.join([RESEARCH_FIELDS.get(f, f) for f in result.get('research_fields', [])])}

{'='*70}
【最適化設定】
  個体数: {config['population_size']}
  世代数: {config['generations']}
  交叉率: {config['crossover_rate']}
  突然変異率: {config['mutation_rate']}

{'='*70}
【最適学生プロファイル】

基本12項目:
"""
    
    for criterion in CRITERIA_NAMES.keys():
        value = profile.get(criterion, 5.5)
        name = CRITERIA_NAMES[criterion]
        report += f"  {name:20s}: {value:5.2f}/10\n"
    
    report += f"\n分野興味:\n"
    field_interests = profile.get("field_interests", {})
    for field, interest in field_interests.items():
        field_name = RESEARCH_FIELDS.get(field, field)
        marker = "★" if field in result.get("research_fields", []) else " "
        report += f"  {marker} {field_name:30s}: {interest:5.2f}/10\n"
    
    fitness_history = result["fitness_history"]
    final_fitness = fitness_history[-1]
    initial_fitness = fitness_history[0]
    convergence_gen = next((i for i, f in enumerate(fitness_history) if f >= final_fitness * 0.99), len(fitness_history))
    
    report += f"""
{'='*70}
【最適化結果】
  初期適合度: {initial_fitness:.4f}
  最終適合度: {final_fitness:.4f}
  改善率: {(final_fitness - initial_fitness) / max(initial_fitness, 0.001) * 100:.2f}%
  収束世代: {convergence_gen}/{config['generations']}

{'='*70}
【特徴分析】

最も重視する項目 (≥ 8.0):
"""
    
    high_priority = [(CRITERIA_NAMES[k], v) for k, v in profile.items() 
                     if k in CRITERIA_NAMES and v >= 8.0]
    high_priority.sort(key=lambda x: x[1], reverse=True)
    
    if high_priority:
        for name, value in high_priority:
            report += f"  • {name}: {value:.2f}\n"
    else:
        report += "  （なし）\n"
    
    report += f"\nあまり重視しない項目 (≤ 3.0):\n"
    low_priority = [(CRITERIA_NAMES[k], v) for k, v in profile.items() 
                    if k in CRITERIA_NAMES and v <= 3.0]
    low_priority.sort(key=lambda x: x[1])
    
    if low_priority:
        for name, value in low_priority:
            report += f"  • {name}: {value:.2f}\n"
    else:
        report += "  （なし）\n"
    
    report += f"\n{'='*70}\n"
    report += f"分析日時: {result['analysis_timestamp']}\n"
    report += f"{'='*70}\n"
    
    # ファイルに保存
    report_path = output_path / f"{lab_id}_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✓ サマリーレポートを保存: {report_path}")
    print(report)


def visualize_single_lab(lab_id: str, output_dir: str = "results/visualizations"):
    """単一研究室の結果を可視化"""
    print(f"\n{'='*70}")
    print(f"研究室 {lab_id} の結果を可視化中...")
    print(f"{'='*70}\n")
    
    output_path = Path(output_dir) / lab_id
    output_path.mkdir(parents=True, exist_ok=True)
    
    result = load_result(lab_id)
    
    # 各種グラフ生成
    plot_fitness_evolution(result, output_path)
    plot_student_profile_radar(result, output_path)
    plot_field_interests(result, output_path)
    generate_summary_report(result, output_path)
    
    print(f"\n{'='*70}")
    print(f"可視化完了: {output_path}")
    print(f"{'='*70}\n")


def visualize_comparison(lab_ids: List[str], output_dir: str = "results/visualizations"):
    """複数研究室の比較可視化"""
    print(f"\n{'='*70}")
    print(f"{len(lab_ids)}研究室の比較可視化中...")
    print(f"{'='*70}\n")
    
    output_path = Path(output_dir) / "comparisons"
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = [load_result(lab_id) for lab_id in lab_ids]
    
    plot_criteria_comparison(results, output_path)
    
    print(f"\n{'='*70}")
    print(f"比較可視化完了: {output_path}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="遺伝的アルゴリズム結果の可視化")
    parser.add_argument("--lab_id", type=str, help="可視化する研究室ID")
    parser.add_argument("--compare", nargs="+", help="比較する研究室IDのリスト")
    parser.add_argument("--all", action="store_true", help="全研究室を可視化")
    parser.add_argument("--output", type=str, default="results/visualizations", help="出力ディレクトリ")
    
    args = parser.parse_args()
    
    if args.compare:
        visualize_comparison(args.compare, args.output)
    elif args.lab_id:
        visualize_single_lab(args.lab_id, args.output)
    elif args.all:
        # 全研究室のファイルを検索
        base_dir = Path("results/genetic_optimization")
        lab_dirs = [d.name for d in base_dir.iterdir() if d.is_dir()]
        
        for lab_id in lab_dirs:
            try:
                visualize_single_lab(lab_id, args.output)
            except Exception as e:
                print(f"エラー: {lab_id} の可視化に失敗しました: {e}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()