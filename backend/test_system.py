#!/usr/bin/env python3
"""
研究室選択支援システム v3.0 - 統合テストスクリプト
適合度計算、ファジィ推論、遺伝的アルゴリズムの動作確認
"""

import sys
import os
import json
import requests
from typing import Dict, Any, List
import time

# カラー出力用
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str):
    """ヘッダー表示"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}")
    print(f"{text}")
    print(f"{'='*70}{Colors.ENDC}\n")

def print_success(text: str):
    """成功メッセージ"""
    print(f"{Colors.GREEN}✅ {text}{Colors.ENDC}")

def print_error(text: str):
    """エラーメッセージ"""
    print(f"{Colors.RED}❌ {text}{Colors.ENDC}")

def print_info(text: str):
    """情報メッセージ"""
    print(f"{Colors.CYAN}ℹ️  {text}{Colors.ENDC}")

def print_warning(text: str):
    """警告メッセージ"""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.ENDC}")

# テスト用学生プロファイル
SAMPLE_STUDENTS = [
    {
        "name": "研究志向型学生",
        "profile": {
            "research_intensity": 9.0,
            "advisor_style": 5.0,
            "team_work": 6.5,
            "workload": 8.5,
            "theory_practice": 4.0,
            "research_field_match": 9.0,
            "skill_development": 7.5,
            "lab_atmosphere": 6.0,
            "flexibility": 5.0,
            "publication_opportunity": 9.5,
            "interdisciplinary": 6.5,
            "communication_style": 6.0,
            "innovation_risk": 8.0
        },
        "field_interests": {
            "人工知能・機械学習": 9.5,
            "画像・映像処理": 7.0,
            "自然科学・数理": 8.0
        }
    },
    {
        "name": "実践志向型学生",
        "profile": {
            "research_intensity": 6.5,
            "advisor_style": 7.5,
            "team_work": 8.5,
            "workload": 7.0,
            "theory_practice": 8.5,
            "research_field_match": 7.5,
            "skill_development": 9.0,
            "lab_atmosphere": 8.5,
            "flexibility": 8.0,
            "publication_opportunity": 6.0,
            "interdisciplinary": 8.0,
            "communication_style": 9.0,
            "innovation_risk": 7.0
        },
        "field_interests": {
            "Webデザイン・UI/UX": 9.0,
            "ゲーム開発・eスポーツ": 8.0,
            "VR/AR・メディアアート": 7.5
        }
    },
    {
        "name": "バランス型学生",
        "profile": {
            "research_intensity": 6.5,
            "advisor_style": 6.5,
            "team_work": 7.0,
            "workload": 6.0,
            "theory_practice": 6.0,
            "research_field_match": 7.5,
            "skill_development": 7.5,
            "lab_atmosphere": 7.0,
            "flexibility": 7.0,
            "publication_opportunity": 7.0,
            "interdisciplinary": 7.0,
            "communication_style": 7.0,
            "innovation_risk": 6.5
        },
        "field_interests": {
            "経営情報・意思決定支援": 7.5,
            "観光情報・地域システム": 7.0,
            "教育・言語学": 6.5
        }
    }
]

class SystemTester:
    """システムテストクラス"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_health_check(self) -> bool:
        """ヘルスチェックテスト"""
        print_header("TEST 1: ヘルスチェック")
        
        try:
            response = self.session.get(f"{self.base_url}/health")
            
            if response.status_code == 200:
                data = response.json()
                print_success("ヘルスチェック成功")
                print_info(f"システムバージョン: {data.get('version')}")
                print_info(f"ステータス: {data.get('status')}")
                
                system_info = data.get('system_info', {})
                print_info(f"研究室数: {system_info.get('labs_count')}")
                print_info(f"評価基準: {system_info.get('evaluation_criteria')}項目")
                print_info(f"研究分野: {system_info.get('research_fields')}分野")
                
                modules = data.get('modules', {})
                print_info(f"モジュール状態:")
                for module, available in modules.items():
                    status = "✅" if available else "❌"
                    print(f"    {status} {module}")
                
                return True
            else:
                print_error(f"ヘルスチェック失敗: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            print_error(f"接続エラー: {e}")
            return False
    
    def test_criteria_endpoint(self) -> bool:
        """評価基準エンドポイントテスト"""
        print_header("TEST 2: 評価基準取得")
        
        try:
            response = self.session.get(f"{self.base_url}/api/criteria")
            
            if response.status_code == 200:
                data = response.json()
                print_success("評価基準取得成功")
                print_info(f"総数: {data.get('total_count')}項目")
                
                categories = data.get('categories', {})
                print_info("カテゴリ別:")
                print(f"    基本項目: {len(categories.get('basic', []))}項目")
                print(f"    拡張項目: {len(categories.get('extended', []))}項目")
                print(f"    特殊項目: {len(categories.get('special', []))}項目")
                
                # サンプル表示
                criteria = data.get('criteria', [])[:3]
                print_info("基準サンプル（先頭3件）:")
                for criterion in criteria:
                    print(f"    • {criterion['name']}: {criterion['description']}")
                
                return True
            else:
                print_error(f"取得失敗: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            print_error(f"エラー: {e}")
            return False
    
    def test_labs_endpoint(self) -> bool:
        """研究室一覧エンドポイントテスト"""
        print_header("TEST 3: 研究室データ取得")
        
        try:
            response = self.session.get(f"{self.base_url}/api/labs")
            
            if response.status_code == 200:
                data = response.json()
                labs = data.get('labs', [])
                
                print_success("研究室データ取得成功")
                print_info(f"研究室総数: {len(labs)}件")
                
                if labs:
                    # サンプル表示
                    print_info(f"研究室サンプル（先頭3件）:")
                    for lab in labs[:3]:
                        print(f"    • {lab.get('name')} ({lab.get('professor')})")
                        print(f"      分野: {', '.join(lab.get('research_fields', [])[:2])}")
                    
                    # 分野統計
                    field_counts = {}
                    for lab in labs:
                        for field in lab.get('research_fields', []):
                            field_counts[field] = field_counts.get(field, 0) + 1
                    
                    print_info(f"分野別研究室数（上位5分野）:")
                    sorted_fields = sorted(field_counts.items(), key=lambda x: x[1], reverse=True)
                    for field, count in sorted_fields[:5]:
                        print(f"    {field}: {count}件")
                
                return len(labs) > 0
            else:
                print_error(f"取得失敗: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            print_error(f"エラー: {e}")
            return False
    
    def test_evaluate_compatibility(self, student: Dict[str, Any]) -> bool:
        """適合度評価テスト"""
        print_header(f"TEST 4: 適合度評価 - {student['name']}")
        
        try:
            # リクエスト構築
            request_data = {
                "student_profile": student["profile"],
                "field_interests": student.get("field_interests", {})
            }
            
            print_info("学生プロファイル:")
            print(f"    研究強度: {student['profile']['research_intensity']}")
            print(f"    チームワーク: {student['profile']['team_work']}")
            print(f"    柔軟性: {student['profile']['flexibility']}")
            
            # API呼び出し
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/api/evaluate",
                json=request_data
            )
            execution_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                
                print_success(f"適合度評価成功（実行時間: {execution_time:.3f}秒）")
                
                # 結果サマリー
                results = data.get('evaluation_results', [])
                top_matches = data.get('top_matches', [])
                
                print_info(f"評価研究室数: {len(results)}件")
                print_info(f"上位マッチング:")
                
                for i, match in enumerate(top_matches[:5], 1):
                    compatibility = match['overall_compatibility']
                    lab_name = match['lab_name']
                    recommendation = match['recommendation_level']
                    
                    # 適合度バー表示
                    bar_length = int(compatibility * 40)
                    bar = "█" * bar_length + "░" * (40 - bar_length)
                    
                    print(f"\n    {i}位: {lab_name}")
                    print(f"        適合度: [{bar}] {compatibility:.1%}")
                    print(f"        推薦レベル: {recommendation}")
                    print(f"        教員: {match['professor']}")
                    
                    # 特徴スコア（上位3項目）
                    feature_scores = match['feature_scores']
                    top_features = sorted(
                        feature_scores.items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )[:3]
                    
                    print(f"        強み:")
                    for feature, score in top_features:
                        print(f"          • {feature}: {score:.2%}")
                
                return True
            else:
                print_error(f"評価失敗: HTTP {response.status_code}")
                print_error(f"レスポンス: {response.text}")
                return False
                
        except Exception as e:
            print_error(f"エラー: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_optimization(self) -> bool:
        """遺伝的アルゴリズム最適化テスト"""
        print_header("TEST 5: 遺伝的アルゴリズム最適化")
        
        try:
            request_data = {
                "training_mode": "balanced",
                "num_samples": 30,  # テスト用に少なめ
                "generations": 20,
                "population_size": 15
            }
            
            print_info("最適化パラメータ:")
            print(f"    訓練データ: {request_data['num_samples']}サンプル")
            print(f"    世代数: {request_data['generations']}")
            print(f"    集団サイズ: {request_data['population_size']}")
            
            print_info("最適化実行中...")
            start_time = time.time()
            
            response = self.session.post(
                f"{self.base_url}/api/optimize",
                json=request_data,
                timeout=300  # 5分タイムアウト
            )
            
            execution_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                
                print_success(f"最適化成功（実行時間: {execution_time:.1f}秒）")
                
                optimal_tree = data.get('optimal_tree', {})
                evolution_summary = data.get('evolution_summary', {})
                
                print_info("最適化結果:")
                print(f"    最終適合度: {optimal_tree.get('fitness', 0):.4f}")
                print(f"    総世代数: {evolution_summary.get('total_generations')}")
                print(f"    収束世代: {evolution_summary.get('convergence_generation')}")
                
                print_info("最適決定木構造:")
                print(f"    レベル1特徴: {optimal_tree.get('level1_feature')}")
                print(f"    レベル2特徴: {', '.join(optimal_tree.get('level2_features', []))}")
                
                return True
            elif response.status_code == 501:
                print_warning("遺伝的アルゴリズムモジュールが利用できません")
                return False
            else:
                print_error(f"最適化失敗: HTTP {response.status_code}")
                return False
                
        except requests.exceptions.Timeout:
            print_error("タイムアウト（最適化に時間がかかりすぎています）")
            return False
        except Exception as e:
            print_error(f"エラー: {e}")
            return False
    
    def run_all_tests(self):
        """全テスト実行"""
        print_header("研究室選択支援システム v3.0 - 統合テスト")
        
        results = []
        
        # Test 1: ヘルスチェック
        results.append(("ヘルスチェック", self.test_health_check()))
        
        # Test 2: 評価基準
        results.append(("評価基準取得", self.test_criteria_endpoint()))
        
        # Test 3: 研究室データ
        results.append(("研究室データ取得", self.test_labs_endpoint()))
        
        # Test 4: 適合度評価（複数の学生タイプ）
        for student in SAMPLE_STUDENTS:
            test_name = f"適合度評価 - {student['name']}"
            results.append((test_name, self.test_evaluate_compatibility(student)))
        
        # Test 5: 最適化（オプション）
        print_info("\n遺伝的アルゴリズムの最適化テストを実行しますか？（時間がかかります）")
        print_info("実行する場合は 'y' を入力してください（スキップする場合はEnter）")
        user_input = input("> ").strip().lower()
        
        if user_input == 'y':
            results.append(("遺伝的アルゴリズム最適化", self.test_optimization()))
        
        # 結果サマリー
        print_header("テスト結果サマリー")
        
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        for test_name, result in results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status}  {test_name}")
        
        print(f"\n{'='*70}")
        print(f"総合結果: {passed}/{total} テスト合格")
        
        if passed == total:
            print_success("🎉 全テスト合格！システムは正常に動作しています。")
        else:
            print_warning(f"⚠️ {total - passed}件のテストが失敗しました。")
        
        print(f"{'='*70}\n")

def main():
    """メイン処理"""
    print(f"{Colors.BOLD}")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║    研究室選択支援システム v3.0 - 統合テストスクリプト             ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.ENDC}")
    
    # サーバーURL入力
    print("\nサーバーURLを入力してください（デフォルト: http://localhost:8000）")
    base_url = input("> ").strip()
    
    if not base_url:
        base_url = "http://localhost:8000"
    
    print_info(f"テスト対象: {base_url}")
    print_info("テストを開始します...\n")
    
    # テスター初期化
    tester = SystemTester(base_url)
    
    # 全テスト実行
    tester.run_all_tests()

if __name__ == "__main__":
    main()