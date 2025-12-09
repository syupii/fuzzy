"""
研究室配属システム - 感度分析モジュール
各研究室が1位になる条件とパラメータ影響度を分析
"""

from typing import Dict, List, Any, Tuple
import numpy as np
from itertools import product
import json

class SensitivityAnalyzer:
    """
    感度分析を実行するクラス
    
    主な機能:
    1. パラメータ影響度分析（各パラメータがスコアに与える影響）
    2. 1位獲得条件分析（各研究室が1位になる条件）
    3. 境界値分析（順位が入れ替わる境界条件）
    """
    
    def __init__(self, matcher, labs_data):
        """
        Args:
            matcher: FuzzyMultiPathMatcher インスタンス
            labs_data: 研究室データベース
        """
        self.matcher = matcher
        self.labs_data = labs_data
        
        # 13の評価基準
        self.criteria = [
            "research_intensity",
            "advisor_style", 
            "team_work",
            "workload",
            "theory_practice",
            "research_field_match",
            "skill_development",
            "lab_atmosphere",
            "flexibility",
            "publication_opportunity",
            "interdisciplinary",
            "communication_style"
        ]
    
    def _normalize_value(self, value: float, min_val: float = 1.0, max_val: float = 10.0) -> float:
        """
        値を0-1スケールに正規化
        
        FuzzyMultiPathMatcher は0-1スケールを期待するため、
        1-10スケールの値を0-1に変換する
        
        Args:
            value: 変換する値
            min_val: 最小値（デフォルト: 1.0）
            max_val: 最大値（デフォルト: 10.0）
            
        Returns:
            0-1スケールの値
        """
        if value < min_val:
            return 0.0
        if value > max_val:
            return 1.0
        return (value - min_val) / (max_val - min_val)
    
    def _normalize_profile(self, profile: Dict) -> Dict:
        """
        学生プロファイル全体を正規化
        
        Args:
            profile: 1-10スケールのプロファイル
            
        Returns:
            0-1スケールのプロファイル
        """
        normalized = {}
        
        for criterion in self.criteria:
            if criterion in profile:
                normalized[criterion] = self._normalize_value(profile[criterion])
            
            # 優先度も正規化
            priority_key = f"{criterion}_priority"
            if priority_key in profile:
                normalized[priority_key] = profile[priority_key]  # 優先度はそのまま
        
        # field_interests はそのままコピー
        if "field_interests" in profile:
            normalized["field_interests"] = profile["field_interests"]
        
        return normalized
    
    def _match_all_labs(self, student_profile: Dict, labs_data: List[Dict]) -> List[Dict]:
        """
        全研究室とのマッチングを実行（ヘルパーメソッド）
        
        FuzzyMultiPathMatcher.calculate_compatibility() は1研究室との
        適合度を計算するので、全研究室についてループで呼び出す
        
        Args:
            student_profile: 学生プロファイル（1-10スケール）
            labs_data: 研究室データベース
            
        Returns:
            各研究室の評価結果（スコア降順）
        """
        # プロファイルを0-1スケールに正規化
        normalized_profile = self._normalize_profile(student_profile)
        
        results = []
        
        for lab in labs_data:
            try:
                # 適合度計算
                result = self.matcher.calculate_compatibility(normalized_profile, lab)
                
                results.append({
                    "lab_id": lab["id"],
                    "lab_name": lab.get("name", lab["id"]),
                    "final_score": result.total_compatibility,
                    "basic_score": result.basic_score,
                    "field_score": result.field_score,
                    "rank": 0  # 後でソート後に設定
                })
            except Exception as e:
                # エラーの場合はスキップ
                print(f"  ⚠️ {lab['id']} の評価エラー: {e}")
                continue
        
        # スコア降順でソート
        results.sort(key=lambda x: x["final_score"], reverse=True)
        
        # 順位を設定
        for i, result in enumerate(results, 1):
            result["rank"] = i
        
        return results
        
    # ==================== Phase 1: パラメータ影響度分析 ====================
    
    def analyze_parameter_importance(self, lab_id: str, base_profile: Dict) -> Dict[str, Any]:
        """
        特定の研究室に対する各パラメータの影響度を分析
        
        Args:
            lab_id: 分析対象の研究室ID
            base_profile: ベースとなる学生プロファイル
            
        Returns:
            各パラメータの影響度スコア
        """
        results = {}
        
        # 対象研究室を取得
        target_lab = next((lab for lab in self.labs_data if lab["id"] == lab_id), None)
        if not target_lab:
            print(f"  ⚠️ 研究室 {lab_id} が見つかりません")
            # 空の結果を返す代わりに、デフォルト構造を返す
            return {
                "lab_id": lab_id,
                "baseline_score": 0.0,
                "parameter_importance": {},
                "top_3_influential": [],
                "least_influential": []
            }
        
        # ベースラインスコアを計算
        normalized_base = self._normalize_profile(base_profile)
        base_result = self.matcher.calculate_compatibility(normalized_base, target_lab)
        baseline_score = base_result.total_compatibility
        
        # 各パラメータを個別に変化させる
        for criterion in self.criteria:
            scores_when_varied = []
            test_values = [1, 3, 5, 7, 10]  # 5段階でテスト
            
            for value in test_values:
                # プロファイルをコピーして1つのパラメータだけ変更
                test_profile = base_profile.copy()
                test_profile[criterion] = value
                test_profile[f"{criterion}_priority"] = 8.0  # 高優先度で固定
                
                # マッチング実行
                test_results = self._match_all_labs(test_profile, self.labs_data)
                test_lab = next((lab for lab in test_results if lab["lab_id"] == lab_id), None)
                
                if test_lab:
                    scores_when_varied.append({
                        "parameter_value": value,
                        "score": test_lab["final_score"],
                        "rank": test_lab.get("rank", 999)
                    })
            
            # 影響度を計算（スコアの変動幅）
            scores = [s["score"] for s in scores_when_varied]
            score_range = max(scores) - min(scores)
            score_std = np.std(scores)
            
            results[criterion] = {
                "baseline_score": baseline_score,
                "score_range": score_range,  # スコアの変動幅
                "score_std": score_std,       # スコアの標準偏差
                "importance": score_range,    # 影響度 = 変動幅
                "score_progression": scores_when_varied
            }
        
        # 重要度でソート
        sorted_criteria = sorted(
            results.items(), 
            key=lambda x: x[1]["importance"], 
            reverse=True
        )
        
        return {
            "lab_id": lab_id,
            "baseline_score": baseline_score,
            "parameter_importance": dict(sorted_criteria),
            "top_3_influential": [c[0] for c in sorted_criteria[:3]],
            "least_influential": [c[0] for c in sorted_criteria[-3:]]
        }
    
    # ==================== Phase 2: 1位獲得条件分析 ====================
    
    def find_top_rank_conditions(
        self, 
        lab_id: str, 
        num_samples: int = 1000
    ) -> Dict[str, Any]:
        """
        特定の研究室が1位になる条件をサンプリングで探索
        
        Args:
            lab_id: 分析対象の研究室ID
            num_samples: サンプリング数（多いほど精度高いが時間かかる）
            
        Returns:
            1位になる条件の統計情報
        """
        top_rank_profiles = []
        
        for _ in range(num_samples):
            # ランダムなプロファイルを生成
            random_profile = self._generate_random_profile()
            
            # マッチング実行
            results = self._match_all_labs(random_profile, self.labs_data)
            
            # この研究室が1位かチェック
            if results and results[0]["lab_id"] == lab_id:
                top_rank_profiles.append(random_profile)
        
        if not top_rank_profiles:
            return {
                "lab_id": lab_id,
                "found_top_rank": False,
                "message": "サンプリング範囲内で1位条件が見つかりませんでした"
            }
        
        # 1位になる条件の統計を計算
        stats = {}
        for criterion in self.criteria:
            values = [p[criterion] for p in top_rank_profiles]
            stats[criterion] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": min(values),
                "max": max(values),
                "median": np.median(values)
            }
        
        return {
            "lab_id": lab_id,
            "found_top_rank": True,
            "num_samples": num_samples,
            "num_top_rank": len(top_rank_profiles),
            "top_rank_probability": len(top_rank_profiles) / num_samples,
            "typical_profile": stats,
            "example_profiles": top_rank_profiles[:3]  # 例を3つ
        }
    
    # ==================== Phase 3: 境界値分析（拡張版） ====================
    
    def find_parameter_boundary(
        self,
        lab_id: str,
        criterion: str,
        base_profile: Dict,
        precision: float = 0.1
    ) -> Dict[str, Any]:
        """
        特定のパラメータについて、1位を獲得/喪失する境界値を探索
        
        Args:
            lab_id: 分析対象の研究室ID
            criterion: 分析するパラメータ
            base_profile: ベースプロファイル
            precision: 探索精度（小さいほど精密）
            
        Returns:
            境界値の情報
        """
        boundaries = {
            "gains_top": None,  # このパラメータがこの値以上で1位獲得
            "loses_top": None   # このパラメータがこの値以下で1位喪失
        }
        
        # 上昇方向の境界を探索（1位を獲得する境界）
        test_profile = base_profile.copy()
        low, high = 1.0, 10.0
        
        while high - low > precision:
            mid = (low + high) / 2
            test_profile[criterion] = mid
            
            results = self._match_all_labs(test_profile, self.labs_data)
            
            if results and results[0]["lab_id"] == lab_id:
                high = mid  # 1位なので、もっと低い値でも1位かもしれない
                boundaries["gains_top"] = mid
            else:
                low = mid   # 1位でないので、もっと高い値が必要
        
        # 下降方向の境界を探索（1位を喪失する境界）
        test_profile[criterion] = 10.0  # 最大値から開始
        results = self._match_all_labs(test_profile, self.labs_data)
        
        if results and results[0]["lab_id"] == lab_id:
            # 10.0で1位の場合、下降境界を探索
            low, high = 1.0, 10.0
            
            while high - low > precision:
                mid = (low + high) / 2
                test_profile[criterion] = mid
                
                results = self._match_all_labs(test_profile, self.labs_data)
                
                if results and results[0]["lab_id"] == lab_id:
                    high = mid  # まだ1位なので、もっと低くできる
                else:
                    low = mid   # 1位を失ったので、これより高い値が必要
                    boundaries["loses_top"] = mid
        
        return {
            "lab_id": lab_id,
            "criterion": criterion,
            "boundaries": boundaries,
            "valid_range": {
                "min": boundaries["loses_top"] if boundaries["loses_top"] else None,
                "max": boundaries["gains_top"] if boundaries["gains_top"] else None
            }
        }
    
    def analyze_parameter_transition_points(
        self,
        lab_id: str,
        criterion: str,
        base_profile: Dict,
        steps: int = 20
    ) -> Dict[str, Any]:
        """
        パラメータを段階的に変化させて、順位とスコアの遷移を詳細に記録
        
        「値をどれくらい変更したら順位が変わったか」を明確にする
        
        Args:
            lab_id: 分析対象の研究室ID
            criterion: 分析するパラメータ
            base_profile: ベースプロファイル
            steps: パラメータを何段階で変化させるか
            
        Returns:
            遷移点の詳細情報
        """
        test_profile = base_profile.copy()
        transitions = []
        previous_rank = None
        previous_score = None
        previous_top_lab = None
        
        # パラメータを1から10まで段階的に変化
        for value in np.linspace(1.0, 10.0, steps):
            test_profile[criterion] = value
            
            results = self._match_all_labs(test_profile, self.labs_data)
            
            # 対象研究室の順位とスコアを取得
            target_lab = next((lab for lab in results if lab["lab_id"] == lab_id), None)
            current_rank = target_lab["rank"] if target_lab else 999
            current_score = target_lab["final_score"] if target_lab else 0.0
            current_top_lab = results[0]["lab_id"] if results else None
            
            # 順位変動を検出
            if previous_rank is not None and current_rank != previous_rank:
                transition = {
                    "parameter_value": value,
                    "previous_value": value - (10.0 - 1.0) / (steps - 1),
                    "rank_change": f"{previous_rank}位 → {current_rank}位",
                    "score_change": current_score - previous_score,
                    "previous_score": previous_score,
                    "current_score": current_score,
                    "new_top_lab": current_top_lab,
                    "lost_to": current_top_lab if current_rank > 1 else None
                }
                transitions.append(transition)
            
            previous_rank = current_rank
            previous_score = current_score
            previous_top_lab = current_top_lab
        
        return {
            "lab_id": lab_id,
            "criterion": criterion,
            "transitions": transitions,
            "total_transitions": len(transitions),
            "most_critical_transition": transitions[0] if transitions else None
        }
    
    def plot_parameter_sensitivity_curve(
        self,
        lab_id: str,
        criterion: str,
        base_profile: Dict,
        output_file: str = None
    ):
        """
        パラメータ1つの変化によるスコア推移をグラフ化
        
        横軸: パラメータ値（1-10）
        縦軸: マッチングスコア
        
        Args:
            lab_id: 分析対象の研究室ID
            criterion: 分析するパラメータ
            base_profile: ベースプロファイル
            output_file: 出力ファイル名
        """
        try:
            import matplotlib.pyplot as plt
            
            test_profile = base_profile.copy()
            param_values = np.linspace(1.0, 10.0, 50)
            scores = []
            ranks = []
            
            for value in param_values:
                test_profile[criterion] = value
                results = self._match_all_labs(test_profile, self.labs_data)
                
                target_lab = next((lab for lab in results if lab["lab_id"] == lab_id), None)
                if target_lab:
                    scores.append(target_lab["final_score"])
                    ranks.append(target_lab["rank"])
                else:
                    scores.append(0.0)
                    ranks.append(999)
            
            # プロット作成
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # 上段: スコア推移
            ax1.plot(param_values, scores, linewidth=2, color='#2E86AB')
            ax1.fill_between(param_values, scores, alpha=0.3, color='#2E86AB')
            ax1.set_ylabel('マッチングスコア', fontsize=12)
            ax1.set_title(f'{lab_id} の感度分析: {criterion}', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # 下段: 順位推移
            ax2.plot(param_values, ranks, linewidth=2, color='#A23B72', marker='o', markersize=3)
            ax2.set_xlabel(f'{criterion} (1: 低 → 10: 高)', fontsize=12)
            ax2.set_ylabel('順位', fontsize=12)
            ax2.invert_yaxis()  # 1位が上になるように反転
            ax2.grid(True, alpha=0.3)
            
            # 1位の領域をハイライト
            rank_1_indices = [i for i, r in enumerate(ranks) if r == 1]
            if rank_1_indices:
                ax2.axvspan(param_values[rank_1_indices[0]], 
                           param_values[rank_1_indices[-1]], 
                           alpha=0.2, color='gold', label='1位獲得領域')
                ax2.legend()
            
            plt.tight_layout()
            
            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                print(f"  ✅ 感度曲線を保存: {output_file}")
            else:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            print(f"  ⚠️ グラフ作成エラー: {e}")
    
    def generate_comprehensive_lab_report(
        self,
        lab_id: str,
        num_samples: int = 500,
        output_dir: str = "lab_reports"
    ) -> Dict[str, Any]:
        """
        特定の研究室について包括的なレポートを生成
        
        卒論に直接記載できる形式で以下を出力:
        1. パラメータ影響度
        2. 1位獲得条件
        3. 境界値
        4. 遷移点の詳細
        5. 感度曲線グラフ
        
        Args:
            lab_id: 分析対象の研究室ID
            num_samples: サンプリング数
            output_dir: 出力ディレクトリ
            
        Returns:
            包括的なレポートデータ
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 研究室名を取得
        lab_name = next(
            (lab["name"] for lab in self.labs_data if lab["id"] == lab_id),
            lab_id
        )
        
        print(f"\n{'='*70}")
        print(f"📊 {lab_name} の包括的分析レポートを生成中...")
        print(f"{'='*70}")
        
        # ベースプロファイル
        base_profile = {criterion: 5.0 for criterion in self.criteria}
        for criterion in self.criteria:
            base_profile[f"{criterion}_priority"] = 5.0
        
        # Phase 1: パラメータ影響度
        print("  [1/5] パラメータ影響度分析...")
        importance = self.analyze_parameter_importance(lab_id, base_profile)
        
        # Phase 2: 1位獲得条件
        print("  [2/5] 1位獲得条件分析...")
        top_conditions = self.find_top_rank_conditions(lab_id, num_samples)
        
        # Phase 3: 境界値
        print("  [3/5] 境界値分析...")
        boundaries = {}
        top_3_params = importance["top_3_influential"]
        
        for criterion in top_3_params:
            boundary = self.find_parameter_boundary(lab_id, criterion, base_profile)
            boundaries[criterion] = boundary
        
        # Phase 4: 遷移点の詳細分析（新機能）
        print("  [4/5] 遷移点の詳細分析...")
        transitions = {}
        
        for criterion in top_3_params:
            transition = self.analyze_parameter_transition_points(
                lab_id, criterion, base_profile, steps=30
            )
            transitions[criterion] = transition
        
        # Phase 5: 感度曲線の可視化（新機能）
        print("  [5/5] 感度曲線を可視化中...")
        for criterion in top_3_params:
            output_file = os.path.join(
                output_dir,
                f"{lab_id}_{criterion}_sensitivity.png"
            )
            self.plot_parameter_sensitivity_curve(
                lab_id, criterion, base_profile, output_file
            )
        
        # レポートを統合
        report = {
            "lab_id": lab_id,
            "lab_name": lab_name,
            "analysis_date": "2025-12-06",
            "phase1_importance": importance,
            "phase2_top_conditions": top_conditions,
            "phase3_boundaries": boundaries,
            "phase4_transitions": transitions,
            "summary": self._generate_detailed_summary(
                lab_name, importance, top_conditions, boundaries, transitions
            )
        }
        
        # JSONとして保存
        import json
        json
    
    # ==================== 包括的分析 ====================
    
    def comprehensive_analysis(
        self, 
        lab_id: str,
        num_samples: int = 500
    ) -> Dict[str, Any]:
        """
        特定の研究室についての包括的な感度分析
        
        Returns:
            Phase 1, 2, 3の結果を統合したレポート
        """
        # ベースプロファイル（全て中間値）
        base_profile = {criterion: 5.0 for criterion in self.criteria}
        for criterion in self.criteria:
            base_profile[f"{criterion}_priority"] = 5.0
        
        # ★★★ 重要：対象研究室の分野に対する興味を設定 ★★★
        # 対象研究室を取得
        target_lab = next((lab for lab in self.labs_data if lab["id"] == lab_id), None)
        if target_lab and target_lab.get("field_id"):
            # その研究室の分野に対する興味を8.0に設定（高い興味）
            lab_field_id = target_lab["field_id"]
            base_profile["field_interests"] = {lab_field_id: 8.0}
            print(f"  📚 分野マッチング有効化: {lab_field_id} = 8.0")
        else:
            # field_idがない場合は空辞書
            base_profile["field_interests"] = {}
            if target_lab:
                print(f"  ⚠️  警告: {lab_id} にはfield_idが設定されていません")
        
        print(f"🔍 {lab_id} の感度分析を開始...")
        
        # Phase 1: パラメータ影響度
        print("  Phase 1: パラメータ影響度分析...")
        importance = self.analyze_parameter_importance(lab_id, base_profile)
        
        # デバッグ: 返り値の構造を確認
        print(f"  📊 Phase 1 結果のキー: {list(importance.keys())}")
        
        # top_3_influential が存在するか確認
        if "top_3_influential" not in importance:
            print(f"  ⚠️ 警告: top_3_influential が見つかりません")
            print(f"  📋 importance の内容: {importance}")
            # デフォルト値を設定
            if "parameter_importance" in importance:
                sorted_params = sorted(
                    importance["parameter_importance"].items(),
                    key=lambda x: x[1].get("importance", 0),
                    reverse=True
                )
                importance["top_3_influential"] = [p[0] for p in sorted_params[:3]]
            else:
                importance["top_3_influential"] = []
        
        # Phase 2: 1位獲得条件
        print("  Phase 2: 1位獲得条件分析...")
        top_conditions = self.find_top_rank_conditions(lab_id, num_samples)
        
        # Phase 3: 主要パラメータの境界値（上位3つのみ）
        print("  Phase 3: 境界値分析...")
        boundaries = {}
        top_3_params = importance["top_3_influential"]
        
        for criterion in top_3_params:
            boundary = self.find_parameter_boundary(lab_id, criterion, base_profile)
            boundaries[criterion] = boundary
        
        return {
            "lab_id": lab_id,
            "analysis_timestamp": "2025-12-06",
            "phase1_parameter_importance": importance,
            "phase2_top_rank_conditions": top_conditions,
            "phase3_boundaries": boundaries,
            "summary": self._generate_summary(importance, top_conditions, boundaries)
        }
    
    def analyze_all_labs(self, num_samples: int = 300) -> Dict[str, Any]:
        """
        全研究室について包括的な感度分析を実行
        
        Args:
            num_samples: 各研究室のサンプリング数
            
        Returns:
            全研究室の分析結果
        """
        all_results = {}
        
        for i, lab in enumerate(self.labs_data, 1):
            lab_id = lab["id"]
            print(f"\n{'='*60}")
            print(f"[{i}/{len(self.labs_data)}] {lab['name']} を分析中...")
            print(f"{'='*60}")
            
            result = self.comprehensive_analysis(lab_id, num_samples)
            all_results[lab_id] = result
        
        # 全体サマリー
        summary = self._generate_global_summary(all_results)
        
        return {
            "total_labs": len(self.labs_data),
            "analysis_date": "2025-12-06",
            "individual_results": all_results,
            "global_summary": summary
        }
    
    # ==================== ヘルパーメソッド ====================
    
    def _generate_random_profile(self) -> Dict[str, float]:
        """ランダムな学生プロファイルを生成"""
        profile = {}
        for criterion in self.criteria:
            profile[criterion] = np.random.uniform(1, 10)
            profile[f"{criterion}_priority"] = np.random.uniform(3, 10)
        
        # field_interests を追加（空でもOK）
        profile["field_interests"] = {}
        
        return profile
    
    def _generate_summary(
        self, 
        importance: Dict, 
        top_conditions: Dict,
        boundaries: Dict
    ) -> Dict[str, Any]:
        """個別研究室のサマリーを生成"""
        return {
            "most_influential_parameters": importance["top_3_influential"],
            "can_achieve_top_rank": top_conditions["found_top_rank"],
            "top_rank_probability": top_conditions.get("top_rank_probability", 0),
            "critical_boundaries_identified": len(boundaries) > 0
        }
    
    def _generate_global_summary(self, all_results: Dict) -> Dict[str, Any]:
        """全体のサマリーを生成"""
        # 最も影響力のあるパラメータを集計
        param_frequency = {criterion: 0 for criterion in self.criteria}
        
        for lab_result in all_results.values():
            top_3 = lab_result["phase1_parameter_importance"]["top_3_influential"]
            for param in top_3:
                param_frequency[param] += 1
        
        # 1位獲得可能な研究室数
        achievable_labs = sum(
            1 for result in all_results.values()
            if result["phase2_top_rank_conditions"]["found_top_rank"]
        )
        
        return {
            "globally_influential_parameters": sorted(
                param_frequency.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
            "labs_with_top_rank_potential": achievable_labs,
            "total_labs_analyzed": len(all_results)
        }
    
    # ==================== エクスポート機能 ====================
    
    def export_to_json(self, results: Dict, filename: str):
        """結果をJSONファイルにエクスポート"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"✅ 結果を {filename} に保存しました")
    
    def export_to_csv_summary(self, results: Dict, filename: str):
        """結果のサマリーをCSVにエクスポート（Excel用）"""
        try:
            import pandas as pd
            
            rows = []
            for lab_id, lab_result in results["individual_results"].items():
                lab_name = next(
                    (lab["name"] for lab in self.labs_data if lab["id"] == lab_id),
                    lab_id
                )
                
                importance = lab_result["phase1_parameter_importance"]
                top_conditions = lab_result["phase2_top_rank_conditions"]
                
                row = {
                    "研究室ID": lab_id,
                    "研究室名": lab_name,
                    "最重要パラメータ1": importance["top_3_influential"][0],
                    "最重要パラメータ2": importance["top_3_influential"][1],
                    "最重要パラメータ3": importance["top_3_influential"][2],
                    "1位獲得可能": "○" if top_conditions["found_top_rank"] else "×",
                    "1位獲得確率": f"{top_conditions.get('top_rank_probability', 0):.2%}"
                }
                
                # 典型プロファイル（1位になる条件）を追加
                if top_conditions["found_top_rank"]:
                    typical = top_conditions["typical_profile"]
                    for criterion in self.criteria:
                        row[f"{criterion}_平均"] = f"{typical[criterion]['mean']:.2f}"
                
                rows.append(row)
            
            df = pd.DataFrame(rows)
            df.to_csv(filename, index=False, encoding='utf-8-sig')  # Excel用BOM付き
            print(f"✅ CSVサマリーを {filename} に保存しました")
            
        except ImportError:
            print("⚠️ pandas が必要です: pip install pandas")
    
    # ==================== 決定境界可視化（モンテカルロ・マッピング） ====================
    
    def plot_decision_boundary_2d(
        self,
        param_x: str,
        param_y: str,
        resolution: int = 50,
        output_file: str = "decision_boundary.png"
    ):
        """
        2つのパラメータについて決定境界を可視化（勢力図）
        
        Args:
            param_x: X軸のパラメータ
            param_y: Y軸のパラメータ
            resolution: グリッドの解像度（高いほど精密だが遅い）
            output_file: 出力ファイル名
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            from matplotlib.colors import ListedColormap
            
            print(f"🗺️ 決定境界を可視化中: {param_x} vs {param_y}")
            
            # グリッド生成
            x_range = np.linspace(1, 10, resolution)
            y_range = np.linspace(1, 10, resolution)
            
            # 結果を格納する配列
            grid_results = np.zeros((resolution, resolution), dtype=int)
            lab_ids = [lab["id"] for lab in self.labs_data]
            lab_id_to_index = {lab_id: i for i, lab_id in enumerate(lab_ids)}
            
            # 各グリッドポイントで1位を計算
            total_points = resolution * resolution
            current = 0
            
            for i, x_val in enumerate(x_range):
                for j, y_val in enumerate(y_range):
                    current += 1
                    if current % 100 == 0:
                        print(f"  進捗: {current}/{total_points} ({current/total_points*100:.1f}%)")
                    
                    # ベースプロファイルを作成（他は中間値5.0）
                    profile = {criterion: 5.0 for criterion in self.criteria}
                    profile[param_x] = x_val
                    profile[param_y] = y_val
                    
                    # 優先度は全て中程度
                    for criterion in self.criteria:
                        profile[f"{criterion}_priority"] = 5.0
                    
                    # マッチング実行
                    results = self._match_all_labs(profile, self.labs_data)
                    
                    if results:
                        top_lab_id = results[0]["lab_id"]
                        grid_results[j, i] = lab_id_to_index.get(top_lab_id, 0)
            
            # プロット作成
            fig, ax = plt.subplots(figsize=(14, 10))
            
            # カラーマップ
            n_labs = len(lab_ids)
            colors = plt.cm.tab20(np.linspace(0, 1, n_labs))
            cmap = ListedColormap(colors)
            
            # コンター図
            im = ax.imshow(
                grid_results,
                extent=[1, 10, 1, 10],
                origin='lower',
                cmap=cmap,
                aspect='auto',
                alpha=0.6
            )
            
            # 境界線を強調
            ax.contour(
                x_range, y_range, grid_results,
                levels=np.arange(0.5, n_labs, 1),
                colors='black',
                linewidths=0.5,
                alpha=0.3
            )
            
            # ラベル
            ax.set_xlabel(f'{param_x} (1: 低 → 10: 高)', fontsize=12)
            ax.set_ylabel(f'{param_y} (1: 低 → 10: 高)', fontsize=12)
            ax.set_title(
                f'決定境界マップ: {param_x} vs {param_y}\n'
                f'各色は異なる研究室が1位になる領域を示す',
                fontsize=14,
                fontweight='bold'
            )
            
            # グリッド
            ax.grid(True, alpha=0.3)
            
            # カラーバー（研究室名）
            cbar = plt.colorbar(im, ax=ax, ticks=range(n_labs))
            lab_names = [lab["name"][:15] for lab in self.labs_data]  # 15文字に短縮
            cbar.ax.set_yticklabels(lab_names, fontsize=7)
            cbar.set_label('研究室', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"✅ 決定境界を保存: {output_file}")
            plt.close()
            
        except ImportError as e:
            print(f"⚠️ 可視化ライブラリが必要です: {e}")
        except Exception as e:
            print(f"❌ 可視化エラー: {e}")
            import traceback
            traceback.print_exc()
    
    def plot_multiple_decision_boundaries(
        self,
        output_dir: str = "decision_boundaries"
    ):
        """
        主要なパラメータペアについて複数の決定境界を生成
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 主要なパラメータペア
        important_pairs = [
            ("research_intensity", "theory_practice"),
            ("research_intensity", "research_field_match"),
            ("theory_practice", "team_work"),
            ("research_field_match", "lab_atmosphere"),
            ("advisor_style", "flexibility"),
        ]
        
        print(f"\n🗺️ {len(important_pairs)}個の決定境界マップを生成します...")
        
        for i, (param_x, param_y) in enumerate(important_pairs, 1):
            print(f"\n[{i}/{len(important_pairs)}] {param_x} vs {param_y}")
            output_file = os.path.join(
                output_dir,
                f"boundary_{param_x}_vs_{param_y}.png"
            )
            self.plot_decision_boundary_2d(param_x, param_y, resolution=40, output_file=output_file)
        
        print(f"\n✅ すべての決定境界マップを保存: {output_dir}/")
    
    def analyze_parameter_space_coverage(self, num_samples: int = 10000) -> Dict:
        """
        パラメータ空間全体での各研究室の「勢力」を分析
        
        Args:
            num_samples: サンプリング数
            
        Returns:
            各研究室が1位になる確率（パラメータ空間全体での占有率）
        """
        print(f"🌐 パラメータ空間の勢力分析（{num_samples}サンプル）...")
        
        lab_win_counts = {lab["id"]: 0 for lab in self.labs_data}
        
        for i in range(num_samples):
            if (i + 1) % 1000 == 0:
                print(f"  進捗: {i+1}/{num_samples}")
            
            # ランダムプロファイル生成
            profile = self._generate_random_profile()
            
            # マッチング実行
            results = self._match_all_labs(profile, self.labs_data)
            
            if results:
                top_lab_id = results[0]["lab_id"]
                lab_win_counts[top_lab_id] += 1
        
        # 確率に変換
        lab_probabilities = {
            lab_id: count / num_samples
            for lab_id, count in lab_win_counts.items()
        }
        
        # ソート
        sorted_labs = sorted(
            lab_probabilities.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        print("\n📊 パラメータ空間の勢力分布:")
        for lab_id, prob in sorted_labs[:10]:  # TOP 10
            lab_name = next(
                (lab["name"] for lab in self.labs_data if lab["id"] == lab_id),
                lab_id
            )
            print(f"  {lab_name}: {prob*100:.2f}%")
        
        return {
            "lab_probabilities": lab_probabilities,
            "sorted_ranking": sorted_labs,
            "total_samples": num_samples
        }


# ==================== FastAPI エンドポイント統合用 ====================

def create_sensitivity_endpoint(app, matcher, labs_data):
    """
    FastAPIアプリケーションに感度分析エンドポイントを追加
    
    使用例:
        from sensitivity_analysis import create_sensitivity_endpoint
        create_sensitivity_endpoint(app, matcher, labs_data)
    """
    from fastapi import Query
    
    analyzer = SensitivityAnalyzer(matcher, labs_data)
    
    @app.post("/api/sensitivity/analyze-lab")
    async def analyze_lab_sensitivity(
        lab_id: str = Query(..., description="研究室ID"),
        num_samples: int = Query(500, description="サンプリング数")
    ):
        """特定の研究室の感度分析"""
        result = analyzer.comprehensive_analysis(lab_id, num_samples)
        return result
    
    @app.post("/api/sensitivity/analyze-all")
    async def analyze_all_labs_sensitivity(
        num_samples: int = Query(300, description="各研究室のサンプリング数")
    ):
        """全研究室の感度分析"""
        results = analyzer.analyze_all_labs(num_samples)
        
        # 結果を自動保存
        analyzer.export_to_json(results, "/home/claude/sensitivity_results.json")
        analyzer.export_to_csv_summary(results, "/home/claude/sensitivity_summary.csv")
        
        return results
    
    @app.get("/api/sensitivity/parameter-importance/{lab_id}")
    async def get_parameter_importance(lab_id: str):
        """パラメータ影響度のみを取得"""
        base_profile = {criterion: 5.0 for criterion in analyzer.criteria}
        for criterion in analyzer.criteria:
            base_profile[f"{criterion}_priority"] = 5.0
        
        result = analyzer.analyze_parameter_importance(lab_id, base_profile)
        return result
    
    print("✅ 感度分析エンドポイントを追加しました")
    print("   - POST /api/sensitivity/analyze-lab")
    print("   - POST /api/sensitivity/analyze-all")
    print("   - GET /api/sensitivity/parameter-importance/{lab_id}")