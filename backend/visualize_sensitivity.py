#!/usr/bin/env python3
"""
感度分析結果の視覚化ツール
卒論用の図表を自動生成

使い方:
  python visualize_sensitivity.py sensitivity_results_20251206_123456.json
  python visualize_sensitivity.py sensitivity_summary_20251206_123456.csv --format csv
"""

import sys
import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# 日本語フォント設定（環境に応じて調整）
plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Yu Gothic', 'Meirio', 'MS Gothic']
plt.rcParams['axes.unicode_minus'] = False

# カラーパレット
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'success': '#06A77D',
    'warning': '#D84315'
}


def load_data(file_path: str, format_type: str = 'json'):
    """データ読み込み"""
    if format_type == 'json':
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif format_type == 'csv':
        return pd.read_csv(file_path)
    else:
        raise ValueError(f"未対応のフォーマット: {format_type}")


def plot_parameter_importance_ranking(data: dict, output_dir: Path):
    """
    図1: パラメータ影響度ランキング
    全体でどのパラメータが最も重要かを示す棒グラフ
    """
    summary = data['global_summary']
    params = summary['globally_influential_parameters'][:10]  # TOP 10
    
    param_names = [p[0] for p in params]
    counts = [p[1] for p in params]
    total_labs = data['total_labs']
    percentages = [(c / total_labs) * 100 for c in counts]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars = ax.barh(param_names, percentages, color=COLORS['primary'])
    
    # 値ラベルを追加
    for i, (bar, pct, cnt) in enumerate(zip(bars, percentages, counts)):
        ax.text(pct + 1, i, f'{pct:.1f}% ({cnt}研究室)', 
                va='center', fontsize=10)
    
    ax.set_xlabel('影響を受ける研究室の割合 (%)', fontsize=12)
    ax.set_ylabel('評価パラメータ', fontsize=12)
    ax.set_title('図1: パラメータ影響度ランキング（全研究室）', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'fig1_parameter_importance_ranking.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 図1を保存: {output_path}")
    plt.close()


def plot_top_rank_probability_distribution(data: dict, output_dir: Path):
    """
    図2: 1位獲得確率の分布
    各研究室が1位になる確率のヒストグラム
    """
    results = data['individual_results']
    
    probabilities = []
    for lab_result in results.values():
        if lab_result['phase2_top_rank_conditions']['found_top_rank']:
            prob = lab_result['phase2_top_rank_conditions']['top_rank_probability'] * 100
            probabilities.append(prob)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    n, bins, patches = ax.hist(probabilities, bins=15, edgecolor='black', 
                                color=COLORS['secondary'], alpha=0.7)
    
    # 統計情報を追加
    mean_prob = np.mean(probabilities)
    median_prob = np.median(probabilities)
    
    ax.axvline(mean_prob, color=COLORS['accent'], linestyle='--', 
               linewidth=2, label=f'平均: {mean_prob:.1f}%')
    ax.axvline(median_prob, color=COLORS['success'], linestyle='--', 
               linewidth=2, label=f'中央値: {median_prob:.1f}%')
    
    ax.set_xlabel('1位獲得確率 (%)', fontsize=12)
    ax.set_ylabel('研究室数', fontsize=12)
    ax.set_title('図2: 1位獲得確率の分布', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'fig2_probability_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 図2を保存: {output_path}")
    plt.close()


def plot_parameter_heatmap(data: dict, output_dir: Path):
    """
    図3: 研究室ごとの典型プロファイルヒートマップ
    各研究室が1位になる条件を可視化
    """
    results = data['individual_results']
    
    # データ準備
    labs = []
    profiles = []
    
    for lab_id, lab_result in results.items():
        if lab_result['phase2_top_rank_conditions']['found_top_rank']:
            labs.append(lab_id)
            typical = lab_result['phase2_top_rank_conditions']['typical_profile']
            
            # 主要パラメータのみ抽出
            profile = {
                'research_intensity': typical['research_intensity']['mean'],
                'advisor_style': typical['advisor_style']['mean'],
                'team_work': typical['team_work']['mean'],
                'theory_practice': typical['theory_practice']['mean'],
                'research_field_match': typical['research_field_match']['mean'],
                'lab_atmosphere': typical['lab_atmosphere']['mean'],
            }
            profiles.append(profile)
    
    if not profiles:
        print("⚠️ ヒートマップ用のデータが不足しています")
        return
    
    # DataFrameに変換
    df = pd.DataFrame(profiles, index=labs)
    
    # 正規化（0-10スケールに戻す）
    df_scaled = df * 9 + 1  # 0-1 → 1-10
    
    # ヒートマップ作成
    fig, ax = plt.subplots(figsize=(10, len(labs) * 0.4))
    
    sns.heatmap(df_scaled, annot=True, fmt='.1f', cmap='YlOrRd', 
                cbar_kws={'label': 'パラメータ値 (1-10)'}, 
                linewidths=0.5, ax=ax)
    
    ax.set_xlabel('評価パラメータ', fontsize=12)
    ax.set_ylabel('研究室ID', fontsize=12)
    ax.set_title('図3: 研究室ごとの典型的な学生プロファイル', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / 'fig3_profile_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 図3を保存: {output_path}")
    plt.close()


def plot_top_3_parameters_per_lab(data: dict, output_dir: Path):
    """
    図4: 各研究室の最重要パラメータTOP3
    研究室ごとの特性を一覧表示
    """
    results = data['individual_results']
    
    # データ準備
    lab_names = []
    top_params = []
    
    for lab_id, lab_result in list(results.items())[:15]:  # 最初の15研究室のみ
        lab_names.append(lab_id)
        importance = lab_result['phase1_parameter_importance']
        top_3 = importance['top_3_influential']
        top_params.append(', '.join(top_3[:2]))  # TOP2のみ表示
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y_pos = np.arange(len(lab_names))
    
    # 表形式で表示
    table_data = [[lab, params] for lab, params in zip(lab_names, top_params)]
    
    table = ax.table(cellText=table_data, 
                     colLabels=['研究室ID', '最重要パラメータ (TOP2)'],
                     cellLoc='left',
                     loc='center',
                     colWidths=[0.3, 0.7])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # ヘッダーのスタイル
    for i in range(2):
        table[(0, i)].set_facecolor(COLORS['primary'])
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 行の色を交互に
    for i in range(1, len(lab_names) + 1):
        color = '#f0f0f0' if i % 2 == 0 else 'white'
        for j in range(2):
            table[(i, j)].set_facecolor(color)
    
    ax.axis('off')
    ax.set_title('図4: 各研究室の最重要パラメータ', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    output_path = output_dir / 'fig4_lab_specific_parameters.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 図4を保存: {output_path}")
    plt.close()


def plot_sensitivity_summary(data: dict, output_dir: Path):
    """
    図5: 感度分析サマリー
    主要な統計情報をまとめたダッシュボード
    """
    summary = data['global_summary']
    total_labs = data['total_labs']
    achievable_labs = summary['labs_with_top_rank_potential']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('図5: 感度分析サマリーダッシュボード', fontsize=16, fontweight='bold')
    
    # (1) 1位獲得可能性
    ax1 = axes[0, 0]
    categories = ['1位獲得可能', '獲得困難']
    sizes = [achievable_labs, total_labs - achievable_labs]
    colors_pie = [COLORS['success'], COLORS['warning']]
    
    wedges, texts, autotexts = ax1.pie(sizes, labels=categories, autopct='%1.1f%%',
                                         colors=colors_pie, startangle=90)
    ax1.set_title('(A) 1位獲得可能性', fontweight='bold')
    
    # (2) パラメータ重要度TOP5
    ax2 = axes[0, 1]
    params = summary['globally_influential_parameters'][:5]
    param_names = [p[0][:20] for p in params]  # 名前を短縮
    counts = [p[1] for p in params]
    
    ax2.bar(range(len(param_names)), counts, color=COLORS['primary'])
    ax2.set_xticks(range(len(param_names)))
    ax2.set_xticklabels(param_names, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('影響を受ける研究室数', fontsize=10)
    ax2.set_title('(B) 最も影響力のあるパラメータ TOP5', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # (3) 1位獲得確率の統計
    ax3 = axes[1, 0]
    probabilities = []
    for lab_result in data['individual_results'].values():
        if lab_result['phase2_top_rank_conditions']['found_top_rank']:
            prob = lab_result['phase2_top_rank_conditions']['top_rank_probability'] * 100
            probabilities.append(prob)
    
    if probabilities:
        stats = {
            '平均': np.mean(probabilities),
            '中央値': np.median(probabilities),
            '最大': np.max(probabilities),
            '最小': np.min(probabilities)
        }
        
        ax3.bar(stats.keys(), stats.values(), color=COLORS['secondary'])
        ax3.set_ylabel('1位獲得確率 (%)', fontsize=10)
        ax3.set_title('(C) 1位獲得確率の統計', fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'データ不足', ha='center', va='center', fontsize=14)
        ax3.set_title('(C) 1位獲得確率の統計', fontweight='bold')
    
    # (4) サマリーテキスト
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = f"""
    📊 分析サマリー
    
    • 分析対象: {total_labs} 研究室
    • 1位獲得可能: {achievable_labs} 研究室
    • 達成率: {(achievable_labs/total_labs)*100:.1f}%
    
    🔥 最重要パラメータ:
    1. {params[0][0]}: {params[0][1]}研究室
    2. {params[1][0]}: {params[1][1]}研究室
    3. {params[2][0]}: {params[2][1]}研究室
    
    💡 平均1位獲得確率: {np.mean(probabilities) if probabilities else 0:.1f}%
    """
    
    ax4.text(0.1, 0.9, summary_text, fontsize=11, verticalalignment='top',
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    output_path = output_dir / 'fig5_summary_dashboard.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 図5を保存: {output_path}")
    plt.close()


def generate_all_figures(data: dict, output_dir: Path):
    """すべての図を生成"""
    print("\n📊 図表を生成中...")
    print("=" * 60)
    
    output_dir.mkdir(exist_ok=True)
    
    try:
        plot_parameter_importance_ranking(data, output_dir)
        plot_top_rank_probability_distribution(data, output_dir)
        plot_parameter_heatmap(data, output_dir)
        plot_top_3_parameters_per_lab(data, output_dir)
        plot_sensitivity_summary(data, output_dir)
        
        print("=" * 60)
        print(f"✅ すべての図表を生成しました: {output_dir}")
        print()
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="感度分析結果の視覚化ツール"
    )
    
    parser.add_argument(
        'input_file',
        type=str,
        help='入力ファイル (JSON or CSV)'
    )
    
    parser.add_argument(
        '--format',
        choices=['json', 'csv'],
        default='json',
        help='入力ファイル形式 (デフォルト: json)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='figures',
        help='出力ディレクトリ (デフォルト: figures)'
    )
    
    args = parser.parse_args()
    
    print("🎨 感度分析結果の視覚化ツール")
    print("=" * 60)
    print(f"📁 入力: {args.input_file}")
    print(f"📁 出力: {args.output}/")
    print("=" * 60)
    
    # データ読み込み
    try:
        data = load_data(args.input_file, args.format)
        print(f"✅ データを読み込みました")
    except Exception as e:
        print(f"❌ データ読み込みエラー: {e}")
        return 1
    
    # 出力ディレクトリ
    output_dir = Path(args.output)
    
    # 図表生成
    generate_all_figures(data, output_dir)
    
    print("✅ すべての処理が完了しました")
    return 0


if __name__ == "__main__":
    sys.exit(main())