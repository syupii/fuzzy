#!/usr/bin/env python3
"""
研究室配属システム - 感度分析実行スクリプト

使い方:
  python run_sensitivity_analysis.py --mode full --samples 500
  python run_sensitivity_analysis.py --mode single --lab lab_ai_ml --samples 300
  python run_sensitivity_analysis.py --mode quick  # 高速モード（全研究室・100サンプル）
"""

import sys
import os
import argparse
import json
from datetime import datetime

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from data.models.labs_database import LabDatabase
    from core.matching.fuzzy_multipath_matcher import FuzzyMultiPathMatcher
    from sensitivity_analysis import SensitivityAnalyzer
except ImportError as e:
    print(f"❌ インポートエラー: {e}")
    print("💡 backend/ ディレクトリから実行してください")
    sys.exit(1)


def print_banner():
    """バナー表示"""
    print("=" * 70)
    print("  🔍 研究室配属システム - 感度分析ツール")
    print("  Version: 1.0.0")
    print("  Date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)
    print()


def initialize_system():
    """システム初期化"""
    print("📚 システムを初期化中...")
    
    # 1. データベース読み込み
    try:
        db = LabDatabase()
        labs_data = db.get_all_labs()
        print(f"  ✅ データベース: {len(labs_data)}研究室を読み込みました")
    except Exception as e:
        print(f"  ❌ データベース読み込みエラー: {e}")
        return None, None, None
    
    # 2. マッチャー初期化
    try:
        matcher = FuzzyMultiPathMatcher()
        print(f"  ✅ マッチャー: 初期化完了")
    except Exception as e:
        print(f"  ❌ マッチャー初期化エラー: {e}")
        return None, None, None
    
    # 3. アナライザー初期化
    try:
        analyzer = SensitivityAnalyzer(matcher, labs_data)
        print(f"  ✅ アナライザー: 初期化完了")
        print()
    except Exception as e:
        print(f"  ❌ アナライザー初期化エラー: {e}")
        return None, None, None
    
    return db, matcher, analyzer


def run_full_analysis(analyzer, num_samples):
    """全研究室の包括的分析"""
    print("🚀 全研究室の感度分析を開始します")
    print(f"  サンプル数: {num_samples}")
    print(f"  推定所要時間: 約{num_samples * len(analyzer.labs_data) // 100}秒")
    print()
    
    start_time = datetime.now()
    
    # 分析実行
    results = analyzer.analyze_all_labs(num_samples=num_samples)
    
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    
    # 結果保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_file = f"sensitivity_results_{timestamp}.json"
    csv_file = f"sensitivity_summary_{timestamp}.csv"
    
    analyzer.export_to_json(results, json_file)
    analyzer.export_to_csv_summary(results, csv_file)
    
    # サマリー表示
    print()
    print("=" * 70)
    print("📊 分析完了サマリー")
    print("=" * 70)
    print(f"✅ 処理時間: {elapsed:.1f}秒")
    print(f"✅ 分析完了: {results['total_labs']}研究室")
    
    summary = results["global_summary"]
    print(f"✅ 1位獲得可能な研究室: {summary['labs_with_top_rank_potential']}/{results['total_labs']}")
    
    print()
    print("🔥 全体で最も影響力のあるパラメータ TOP 5:")
    for i, (param, count) in enumerate(summary["globally_influential_parameters"][:5], 1):
        percentage = (count / results['total_labs']) * 100
        print(f"  {i}. {param}: {count}研究室 ({percentage:.1f}%)")
    
    print()
    print("💾 出力ファイル:")
    print(f"  📄 {json_file} - 詳細データ（JSON）")
    print(f"  📊 {csv_file} - サマリー（CSV / Excel用）")
    print()
    
    return results


def run_single_lab_analysis(analyzer, lab_id, num_samples):
    """特定の研究室の詳細分析"""
    print(f"🎯 研究室 '{lab_id}' の感度分析を開始します")
    print(f"  サンプル数: {num_samples}")
    print()
    
    start_time = datetime.now()
    
    # 分析実行
    result = analyzer.comprehensive_analysis(lab_id, num_samples)
    
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    
    # 結果表示
    print()
    print("=" * 70)
    print(f"📊 '{lab_id}' の分析結果")
    print("=" * 70)
    print(f"✅ 処理時間: {elapsed:.1f}秒")
    print()
    
    # Phase 1: パラメータ影響度
    importance = result["phase1_parameter_importance"]
    print("📌 Phase 1: パラメータ影響度 TOP 5")
    top_5 = importance["top_3_influential"][:5]
    for i, param in enumerate(top_5, 1):
        impact = importance["parameter_importance"][param]["importance"]
        print(f"  {i}. {param}: {impact:.3f}")
    print()
    
    # Phase 2: 1位獲得条件
    top_conditions = result["phase2_top_rank_conditions"]
    if top_conditions["found_top_rank"]:
        prob = top_conditions["top_rank_probability"] * 100
        print(f"📌 Phase 2: 1位獲得条件")
        print(f"  ✅ 1位獲得確率: {prob:.1f}%")
        print(f"  サンプル: {top_conditions['num_top_rank']}/{top_conditions['num_samples']}")
        print()
        
        print("  典型的な学生プロファイル（1位になる条件）:")
        typical = top_conditions["typical_profile"]
        
        # 重要なパラメータのみ表示（TOP 5）
        for param in top_5:
            if param in typical:
                mean = typical[param]["mean"]
                std = typical[param]["std"]
                min_val = typical[param]["min"]
                max_val = typical[param]["max"]
                print(f"    {param}: 平均{mean:.2f} ±{std:.2f} (範囲: {min_val:.1f}-{max_val:.1f})")
    else:
        print(f"📌 Phase 2: 1位獲得条件")
        print(f"  ⚠️ サンプリング範囲内で1位条件が見つかりませんでした")
        print(f"  💡 より多くのサンプルを試すか、パラメータ範囲を確認してください")
    print()
    
    # Phase 3: 境界値
    boundaries = result["phase3_boundaries"]
    if boundaries:
        print("📌 Phase 3: 重要パラメータの境界値")
        for param, boundary_data in boundaries.items():
            bounds = boundary_data["boundaries"]
            if bounds["gains_top"] or bounds["loses_top"]:
                print(f"  {param}:")
                if bounds["gains_top"]:
                    print(f"    → {bounds['gains_top']:.2f}以上で1位獲得")
                if bounds["loses_top"]:
                    print(f"    → {bounds['loses_top']:.2f}以下で1位喪失")
    
    print()
    
    # 結果保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_file = f"sensitivity_{lab_id}_{timestamp}.json"
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"💾 詳細結果を保存: {json_file}")
    print()
    
    return result


def run_quick_mode(analyzer):
    """高速モード（全研究室・少サンプル）"""
    print("⚡ 高速モードで実行します")
    print("  サンプル数: 100（精度低・速度優先）")
    print()
    
    return run_full_analysis(analyzer, num_samples=100)


def list_labs(db):
    """研究室一覧を表示"""
    labs = db.get_all_labs()
    
    print()
    print("=" * 70)
    print("📚 利用可能な研究室一覧")
    print("=" * 70)
    
    for i, lab in enumerate(labs, 1):
        print(f"{i:2d}. {lab['id']:20s} - {lab['name']}")
    
    print()
    print(f"合計: {len(labs)}研究室")
    print()


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description="研究室配属システム - 感度分析ツール",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 全研究室を分析（標準）
  python run_sensitivity_analysis.py --mode full --samples 500

  # 特定の研究室を分析
  python run_sensitivity_analysis.py --mode single --lab lab_ai_ml --samples 300

  # 高速モード（全研究室・100サンプル）
  python run_sensitivity_analysis.py --mode quick

  # 研究室一覧を表示
  python run_sensitivity_analysis.py --list
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['full', 'single', 'quick'],
        default='full',
        help='分析モード (デフォルト: full)'
    )
    
    parser.add_argument(
        '--lab',
        type=str,
        help='分析対象の研究室ID (mode=single の場合に必須)'
    )
    
    parser.add_argument(
        '--samples',
        type=int,
        default=300,
        help='サンプリング数 (デフォルト: 300)'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='研究室一覧を表示して終了'
    )
    
    args = parser.parse_args()
    
    # バナー表示
    print_banner()
    
    # システム初期化
    db, matcher, analyzer = initialize_system()
    
    if not analyzer:
        print("❌ システム初期化に失敗しました")
        return 1
    
    # 研究室一覧表示モード
    if args.list:
        list_labs(db)
        return 0
    
    # モード別実行
    try:
        if args.mode == 'quick':
            run_quick_mode(analyzer)
        
        elif args.mode == 'single':
            if not args.lab:
                print("❌ エラー: --lab オプションで研究室IDを指定してください")
                print("💡 研究室一覧: python run_sensitivity_analysis.py --list")
                return 1
            run_single_lab_analysis(analyzer, args.lab, args.samples)
        
        elif args.mode == 'full':
            run_full_analysis(analyzer, args.samples)
        
        print("✅ すべての処理が完了しました")
        return 0
    
    except KeyboardInterrupt:
        print("\n⚠️ ユーザーによって中断されました")
        return 1
    
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())