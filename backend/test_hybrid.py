#!/usr/bin/env python3
"""
ハイブリッドシステム（ファジィ決定木 × 遺伝的アルゴリズム）テストスクリプト
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def print_section(title):
    """セクションヘッダー"""
    print(f"\n{'='*70}")
    print(f"🔬 {title}")
    print('='*70)

def test_1_health_check():
    """テスト1: システム状態確認"""
    print_section("テスト1: システム状態確認")
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ ヘルスチェック成功")
            print(f"  システム初期化: {data.get('system_initialized')}")
            print(f"  遺伝的アルゴリズム: {data.get('modules', {}).get('genetic')}")
            print(f"  決定木: {data.get('modules', {}).get('decision_tree')}")
            return True
        else:
            print(f"❌ ヘルスチェック失敗: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ エラー: {e}")
        return False

def test_2_train_hybrid_system():
    """テスト2: ハイブリッドシステムの学習"""
    print_section("テスト2: ハイブリッドシステムの学習")
    
    print(f"📊 サンプル学生データで学習を実行中...")
    print(f"  ⏱️  30-60秒ほどかかります...")
    
    try:
        start_time = time.time()
        
        # 空のリクエストでサンプルデータを自動生成
        response = requests.post(
            f"{BASE_URL}/api/hybrid/train",
            json={},
            timeout=120
        )
        
        training_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"\n✅ ハイブリッドシステム学習成功")
            print(f"  処理時間: {training_time:.2f}秒")
            print(f"  学習時間: {data.get('training_time', 0):.2f}秒")
            
            clusters = data.get('clusters', {})
            print(f"\n  📊 学生クラスタリング結果:")
            for cluster_name, count in clusters.items():
                print(f"    {cluster_name}: {count}名")
            
            training_result = data.get('training_result', {})
            if 'cluster_weights' in training_result:
                print(f"\n  🧬 クラスタ別最適重み:")
                for cluster, weights in training_result['cluster_weights'].items():
                    print(f"    {cluster}: {[f'{w:.3f}' for w in weights[:5]]}...")
            
            return True
        else:
            print(f"❌ 学習失敗: HTTP {response.status_code}")
            print(f"  詳細: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"❌ タイムアウト: 学習に時間がかかりすぎています")
        return False
    except Exception as e:
        print(f"❌ エラー: {e}")
        return False

def test_3_evaluate_with_hybrid():
    """テスト3: ハイブリッドシステムによる評価"""
    print_section("テスト3: ハイブリッドシステムによる評価")
    
    # テストプロファイル（高研究強度の学生）
    test_profile = {
        "student_profile": {
            "research_intensity": 0.85,  # 高研究強度
            "advisor_style": 0.7,
            "team_work": 0.65,
            "workload": 0.8,
            "theory_practice": 0.75
        }
    }
    
    print(f"📊 高研究強度学生のプロファイルで評価中...")
    print(f"  research_intensity: {test_profile['student_profile']['research_intensity']}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/hybrid/evaluate",
            json=test_profile,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            
            lab_results = data.get("lab_results", [])
            summary = data.get("summary", {})
            metadata = data.get("metadata", {})
            
            print(f"\n✅ ハイブリッド評価成功")
            print(f"  評価方法: {metadata.get('evaluation_method')}")
            print(f"  決定木使用: {metadata.get('decision_tree_used')}")
            print(f"  クラスタ最適化: {metadata.get('cluster_based_optimization')}")
            
            print(f"\n  📊 統計:")
            print(f"    研究室数: {summary.get('total_labs')}")
            print(f"    平均スコア: {summary.get('avg_score', 0):.3f}")
            print(f"    最高スコア: {summary.get('max_score', 0):.3f}")
            
            if lab_results:
                print(f"\n  🏆 トップ5研究室:")
                for i, lab in enumerate(lab_results[:5], 1):
                    cluster = lab.get('cluster', 'unknown')
                    score = lab.get('final_score', 0)
                    print(f"    {i}. {lab.get('lab_name')} "
                          f"[{cluster}] - {score:.3f}")
            
            return True
        elif response.status_code == 400:
            print(f"❌ エラー: システムが学習されていません")
            print(f"  先に /api/hybrid/train を実行してください")
            return False
        else:
            print(f"❌ 評価失敗: HTTP {response.status_code}")
            print(f"  詳細: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ エラー: {e}")
        return False

def test_4_compare_methods():
    """テスト4: 評価方法の比較"""
    print_section("テスト4: 評価方法の比較")
    
    test_profile = {
        "student_profile": {
            "research_intensity": 0.75,
            "advisor_style": 0.7,
            "team_work": 0.7,
            "workload": 0.75,
            "theory_practice": 0.8
        }
    }
    
    print(f"📊 同じプロファイルで3つの方法を比較:")
    print(f"  1. 基本評価（デフォルト重み）")
    print(f"  2. GA最適化評価")
    print(f"  3. ハイブリッド評価（決定木 + GA）")
    
    results = {}
    
    # 基本評価
    try:
        response = requests.post(
            f"{BASE_URL}/api/evaluate",
            json=test_profile,
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            top_lab = data.get("lab_results", [{}])[0]
            results["基本評価"] = {
                "lab": top_lab.get("lab_name", "N/A"),
                "score": top_lab.get("final_score", 0)
            }
            print(f"\n  ✅ 基本評価完了")
    except:
        print(f"\n  ❌ 基本評価失敗")
    
    # ハイブリッド評価
    try:
        response = requests.post(
            f"{BASE_URL}/api/hybrid/evaluate",
            json=test_profile,
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            top_lab = data.get("lab_results", [{}])[0]
            results["ハイブリッド評価"] = {
                "lab": top_lab.get("lab_name", "N/A"),
                "score": top_lab.get("final_score", 0),
                "cluster": top_lab.get("cluster", "N/A")
            }
            print(f"  ✅ ハイブリッド評価完了")
    except:
        print(f"  ❌ ハイブリッド評価失敗（学習が必要）")
    
    # 結果比較
    if results:
        print(f"\n  📊 比較結果:")
        for method, result in results.items():
            print(f"\n    [{method}]")
            print(f"      トップ研究室: {result['lab']}")
            print(f"      スコア: {result['score']:.4f}")
            if 'cluster' in result:
                print(f"      分類クラスタ: {result['cluster']}")
        
        # スコア差分
        if "基本評価" in results and "ハイブリッド評価" in results:
            diff = results["ハイブリッド評価"]["score"] - results["基本評価"]["score"]
            improvement = (diff / results["基本評価"]["score"]) * 100 if results["基本評価"]["score"] > 0 else 0
            print(f"\n  💡 ハイブリッド評価の改善:")
            print(f"    スコア差: {diff:+.4f}")
            print(f"    改善率: {improvement:+.2f}%")
    
    return len(results) > 0

def run_all_tests():
    """全テスト実行"""
    print("\n" + "="*70)
    print("🔬 ハイブリッドシステム 統合テスト")
    print("="*70)
    print(f"対象サーバー: {BASE_URL}")
    
    results = {}
    
    # テスト実行
    results["システム状態確認"] = test_1_health_check()
    
    if not results["システム状態確認"]:
        print(f"\n❌ サーバーが起動していないため、テストを中止します")
        print(f"  サーバーを起動してください: cd backend && python app.py")
        return
    
    time.sleep(0.5)
    results["ハイブリッドシステム学習"] = test_2_train_hybrid_system()
    
    if not results["ハイブリッドシステム学習"]:
        print(f"\n⚠️  学習が失敗したため、評価テストはスキップします")
    else:
        time.sleep(0.5)
        results["ハイブリッド評価"] = test_3_evaluate_with_hybrid()
        
        time.sleep(0.5)
        results["評価方法比較"] = test_4_compare_methods()
    
    # 結果サマリー
    print("\n" + "="*70)
    print("📊 テスト結果サマリー")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    all_passed = all(results.values())
    print(f"\n総合結果: {'✅ 全テスト合格' if all_passed else '❌ 一部テスト失敗'}")
    
    if all_passed:
        print("\n🎉 ハイブリッドシステムは正しく動作しています！")
        print("\n📖 使い方:")
        print("  1. POST /api/hybrid/train でシステムを学習")
        print("  2. POST /api/hybrid/evaluate で評価")
        print("  3. ファジィ決定木で分類 → GAで最適化された重みで評価")
    else:
        print("\n⚠️  一部の機能に問題があります。")

if __name__ == "__main__":
    try:
        run_all_tests()
    except KeyboardInterrupt:
        print("\n\n⚠️  テストが中断されました")
    except Exception as e:
        print(f"\n❌ 予期しないエラー: {e}")
        import traceback
        traceback.print_exc()