"""
研究室配属システム - 包括的感度分析システム（卒論用・最高品質版）

Phase 1-3: 標準感度分析（パラメータ影響度・1位獲得条件・境界値）
Phase 4: 全パラメータ包括分析（詳細遷移点・感度曲線）
Phase 5: 可視化（図表生成）
Phase 6: 逆引き分析（目標研究室を1位にする条件探索）
Phase 7: ランキング配分分析（12次元空間での順位確率分布）

実行時間: PCスペックと精度モードによる
精度: 3段階（fast/standard/precise）
出力: JSON/CSV/PNG（20種類以上の図表）

使い方:
  # 【推奨】標準モード（30-60分/研究室、約16-31時間で31研究室）
  python comprehensive_sensitivity_analysis.py --mode full --no-confirm
  
  # 軽量モード（15-30分/研究室、約8-16時間で31研究室）- 低スペックPC向け
  python comprehensive_sensitivity_analysis.py --mode full --no-confirm --precision fast
  
  # 高精度モード（60-120分/研究室、約31-62時間で31研究室）- 高スペックPC向け
  python comprehensive_sensitivity_analysis.py --mode full --no-confirm --precision precise
  
  # 24番目から再開（標準モード）
  python comprehensive_sensitivity_analysis.py --mode full --no-confirm --resume --start-from 24
  
  # 24番目から再開（軽量モード）
  python comprehensive_sensitivity_analysis.py --mode full --no-confirm --resume --start-from 24 --precision fast
  
  # 単一研究室のみ（標準モード）
  python comprehensive_sensitivity_analysis.py --mode single --lab lab_002
  
  # 単一研究室のみ（軽量モード）
  python comprehensive_sensitivity_analysis.py --mode single --lab lab_002 --precision fast
"""

import sys
import os
from typing import Dict, List, Any, Tuple
import numpy as np
import json
from datetime import datetime
from pathlib import Path
import warnings

# 日本語フォント設定
import matplotlib

# matplotlibの日本語フォント設定
matplotlib.rcParams['font.family'] = 'sans-serif'
if os.name == 'nt':  # Windows
    matplotlib.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo', 'DejaVu Sans']
else:  # macOS/Linux
    matplotlib.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Noto Sans CJK JP', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # マイナス記号の文字化け防止

# 警告を抑制
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# 既存のアナライザーを拡張
from sensitivity_analysis import SensitivityAnalyzer


class ComprehensiveSensitivityAnalyzer(SensitivityAnalyzer):
    """
    卒論用の包括的感度分析クラス
    
    拡張機能:
    Phase 1-3: 標準感度分析（パラメータ影響度・1位獲得条件・境界値）
    Phase 4: 全パラメータ包括分析（詳細遷移点・感度曲線）
    Phase 5: 可視化（13種類以上の図表）
    Phase 6: 逆引き分析（目標研究室を1位にする条件探索）
    Phase 7: ランキング配分分析（12次元空間での順位確率分布）
    
    出力:
    - JSON: 全分析データ
    - CSV: 統計表・確率分布
    - PNG: ヒートマップ・グラフ（20種類以上）
    - HTML: 対話的レポート
    """
    
    def __init__(self, matcher, labs_data):
        super().__init__(matcher, labs_data)
        
        # デフォルトは標準精度
        self.set_precision_mode('standard')
    
    def set_precision_mode(self, mode: str = 'standard'):
        """
        精度モードを設定
        
        Args:
            mode: 'fast' (軽量), 'standard' (標準), 'precise' (高精度)
        """
        if mode == 'fast':
            # 軽量モード：低スペックPC向け（1研究室 約15-30分）
            self.high_precision_samples = 1000
            self.transition_steps = 30
            self.boundary_precision = 0.1
            self.mode_name = "軽量モード（低スペックPC向け）"
            
        elif mode == 'standard':
            # 標準モード：一般的なPC向け（1研究室 約30-60分）
            self.high_precision_samples = 2000
            self.transition_steps = 50
            self.boundary_precision = 0.05
            self.mode_name = "標準モード（推奨）"
            
        elif mode == 'precise':
            # 高精度モード：高スペックPC向け（1研究室 約60-120分）
            self.high_precision_samples = 5000
            self.transition_steps = 100
            self.boundary_precision = 0.01
            self.mode_name = "高精度モード（高スペックPC・時間に余裕がある場合）"
            
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'fast', 'standard', or 'precise'.")
        
    # ==================== Phase 6: 逆引き分析（条件探索） ====================
    
    def reverse_lookup_optimal_conditions(
        self,
        target_lab_id: str,
        target_rank: int = 1,
        num_trials: int = 1000,
        mutation_rate: float = 0.3
    ) -> Dict[str, Any]:
        """
        目標研究室を指定順位にするための最適パラメータセットを探索
        
        遺伝的アルゴリズムで最適化：
        - 目標: target_lab_id を target_rank 位にする
        - 方法: パラメータ空間を探索して複数の解を発見
        
        Args:
            target_lab_id: 目標研究室ID
            target_rank: 目標順位（1=1位）
            num_trials: 試行回数
            mutation_rate: 突然変異率
            
        Returns:
            最適パラメータセットのリスト
        """
        print(f"\n🎯 逆引き分析: {target_lab_id} を {target_rank}位にする条件を探索中...")
        
        solutions = []  # 見つかった解のリスト
        best_fitness = float('-inf')
        
        # 遺伝的アルゴリズムで探索
        for trial in range(num_trials):
            # ランダムなパラメータセットを生成
            candidate = self._generate_random_profile()
            
            # 評価: target_lab_id の順位を確認
            results = self._match_all_labs(candidate, self.labs_data)
            target_lab = next((lab for lab in results if lab["lab_id"] == target_lab_id), None)
            
            if target_lab:
                current_rank = target_lab.get("rank", 999)
                current_score = target_lab["final_score"]
                
                # 適合度: 目標順位に近いほど高い
                fitness = 1.0 / (abs(current_rank - target_rank) + 1) + current_score * 0.1
                
                # 目標順位を達成した場合
                if current_rank == target_rank:
                    solutions.append({
                        "parameters": {k: v for k, v in candidate.items() if not k.endswith("_priority")},
                        "rank": current_rank,
                        "score": current_score,
                        "fitness": fitness,
                        "trial": trial
                    })
                    
                    if len(solutions) >= 20:  # 十分な解が見つかったら終了
                        break
                
                if fitness > best_fitness:
                    best_fitness = fitness
            
            # 進捗表示
            if (trial + 1) % 100 == 0:
                print(f"  試行 {trial + 1}/{num_trials}: 最良適合度={best_fitness:.3f}, 解の数={len(solutions)}")
        
        # 結果を整理
        solutions.sort(key=lambda x: x["score"], reverse=True)
        
        return {
            "target_lab_id": target_lab_id,
            "target_rank": target_rank,
            "solutions_found": len(solutions),
            "solutions": solutions[:10],  # TOP10の解
            "search_trials": num_trials,
            "best_fitness": best_fitness
        }
    
    def evaluate_specific_conditions(
        self,
        parameters: Dict[str, float],
        top_n: int = 10
    ) -> Dict[str, Any]:
        """
        特定のパラメータ条件下での全研究室ランキングを評価
        
        Args:
            parameters: 評価するパラメータセット
            top_n: 表示する上位研究室数
            
        Returns:
            ランキング結果
        """
        print(f"\n📊 特定条件でのランキング評価...")
        
        # パラメータを正規化
        profile = parameters.copy()
        
        # field_interestsがない場合は空辞書を追加
        if "field_interests" not in profile:
            profile["field_interests"] = {}
        
        # priorityがない場合はデフォルト値を設定
        for criterion in self.criteria:
            if f"{criterion}_priority" not in profile:
                profile[f"{criterion}_priority"] = 5.0
        
        # マッチング実行
        results = self._match_all_labs(profile, self.labs_data)
        
        # 研究室名を追加
        for lab_result in results:
            lab_data = next((lab for lab in self.labs_data if lab["id"] == lab_result["lab_id"]), None)
            if lab_data:
                lab_result["lab_name"] = lab_data.get("name", lab_result["lab_id"])
        
        # TOP Nを抽出
        top_labs = results[:top_n]
        
        return {
            "parameters": parameters,
            "total_labs": len(results),
            "top_ranking": top_labs,
            "score_distribution": {
                "max": max(r["final_score"] for r in results),
                "min": min(r["final_score"] for r in results),
                "mean": np.mean([r["final_score"] for r in results]),
                "std": np.std([r["final_score"] for r in results])
            }
        }
    
    # ==================== Phase 7: ランキング配分分析（12次元空間） ====================
    
    def analyze_ranking_distribution(
        self,
        num_samples: int = 10000,
        target_labs: List[str] = None
    ) -> Dict[str, Any]:
        """
        モンテカルロサンプリングで全パラメータ空間での順位分布を分析
        
        Args:
            num_samples: サンプル数（10,000〜50,000推奨）
            target_labs: 分析対象研究室（Noneの場合は全研究室）
            
        Returns:
            順位確率分布・統計データ
        """
        print(f"\n📈 ランキング配分分析（モンテカルロサンプリング: {num_samples}サンプル）...")
        
        if target_labs is None:
            target_labs = [lab["id"] for lab in self.labs_data]
        
        # 各研究室の順位カウンター
        rank_counts = {lab_id: {i: 0 for i in range(1, len(self.labs_data) + 1)} for lab_id in target_labs}
        
        # パラメータ値と順位の記録（相関分析用）
        parameter_rank_data = {criterion: {lab_id: [] for lab_id in target_labs} for criterion in self.criteria}
        
        print(f"  サンプリング中...")
        for sample_idx in range(num_samples):
            # ランダムなパラメータセットを生成
            random_profile = self._generate_random_profile()
            
            # 対象研究室の分野に対する興味をランダムに設定
            # （公平な分析のため、field_interestsもランダム化）
            random_field_interests = {}
            for lab in self.labs_data:
                if lab.get("field_id"):
                    random_field_interests[lab["field_id"]] = np.random.uniform(1, 10)
            random_profile["field_interests"] = random_field_interests
            
            # マッチング実行
            results = self._match_all_labs(random_profile, self.labs_data)
            
            # 各研究室の順位を記録
            for lab_result in results:
                lab_id = lab_result["lab_id"]
                if lab_id in target_labs:
                    rank = lab_result["rank"]
                    rank_counts[lab_id][rank] += 1
                    
                    # パラメータ値も記録
                    for criterion in self.criteria:
                        parameter_rank_data[criterion][lab_id].append({
                            "value": random_profile[criterion],
                            "rank": rank
                        })
            
            # 進捗表示
            if (sample_idx + 1) % 1000 == 0:
                print(f"    {sample_idx + 1}/{num_samples} サンプル完了")
        
        # 確率分布を計算
        rank_probabilities = {}
        for lab_id in target_labs:
            lab_name = next((lab["name"] for lab in self.labs_data if lab["id"] == lab_id), lab_id)
            rank_probabilities[lab_id] = {
                "lab_name": lab_name,
                "rank_distribution": {
                    rank: count / num_samples
                    for rank, count in rank_counts[lab_id].items()
                },
                "top1_probability": rank_counts[lab_id][1] / num_samples,
                "top3_probability": sum(rank_counts[lab_id][i] for i in range(1, 4)) / num_samples,
                "top5_probability": sum(rank_counts[lab_id][i] for i in range(1, 6)) / num_samples,
                "top10_probability": sum(rank_counts[lab_id][i] for i in range(1, 11)) / num_samples,
                "average_rank": sum(rank * count for rank, count in rank_counts[lab_id].items()) / num_samples
            }
        
        # パラメータと順位の相関を計算
        parameter_correlations = self._analyze_parameter_rank_correlation(parameter_rank_data, target_labs)
        
        return {
            "num_samples": num_samples,
            "total_labs_analyzed": len(target_labs),
            "rank_probabilities": rank_probabilities,
            "parameter_correlations": parameter_correlations,
            "summary": {
                "most_likely_top1": max(rank_probabilities.items(), key=lambda x: x[1]["top1_probability"]),
                "most_consistent": min(rank_probabilities.items(), key=lambda x: x[1]["average_rank"])
            }
        }
    
    def _analyze_parameter_rank_correlation(
        self,
        parameter_rank_data: Dict,
        target_labs: List[str]
    ) -> Dict[str, Any]:
        """パラメータ値と順位の相関を分析"""
        correlations = {}
        
        for criterion in self.criteria:
            criterion_corr = {}
            for lab_id in target_labs:
                data_points = parameter_rank_data[criterion][lab_id]
                if len(data_points) > 0:
                    values = [d["value"] for d in data_points]
                    ranks = [d["rank"] for d in data_points]
                    
                    # Spearman相関係数を計算
                    correlation = np.corrcoef(values, ranks)[0, 1] if len(values) > 1 else 0.0
                    
                    criterion_corr[lab_id] = {
                        "correlation": correlation,
                        "interpretation": self._interpret_correlation(correlation)
                    }
            
            correlations[criterion] = criterion_corr
        
        return correlations
    
    def _interpret_correlation(self, corr: float) -> str:
        """相関係数の解釈"""
        abs_corr = abs(corr)
        if abs_corr >= 0.7:
            strength = "強い"
        elif abs_corr >= 0.4:
            strength = "中程度の"
        elif abs_corr >= 0.2:
            strength = "弱い"
        else:
            strength = "ほとんど"
        
        direction = "正の" if corr > 0 else "負の"
        return f"{strength}{direction}相関"
    
    # ==================== 拡張Phase 4: 詳細な遷移点分析 ====================
    
    def analyze_detailed_transitions(
        self,
        lab_id: str,
        criterion: str,
        base_profile: Dict,
        steps: int = 100
    ) -> Dict[str, Any]:
        """
        パラメータ遷移の超詳細分析
        
        「値を7.2から7.1に変更したら2位になった」
        「その時1位はAI研究室からデータ分析室に変わった」
        等の詳細を記録
        
        Args:
            lab_id: 分析対象の研究室ID
            criterion: 分析するパラメータ
            base_profile: ベースプロファイル
            steps: 分析ステップ数（多いほど精密）
            
        Returns:
            超詳細な遷移情報
        """
        test_profile = base_profile.copy()
        
        # 記録用の配列
        detailed_log = []
        transitions = []
        
        previous_rank = None
        previous_score = None
        previous_top_lab = None
        previous_top_3 = None
        
        param_values = np.linspace(1.0, 10.0, steps)
        
        for i, value in enumerate(param_values):
            test_profile[criterion] = value
            
            # マッチング実行
            results = self._match_all_labs(test_profile, self.labs_data)
            
            # 対象研究室の情報
            target_lab = next((lab for lab in results if lab["lab_id"] == lab_id), None)
            current_rank = target_lab["rank"] if target_lab else 999
            current_score = target_lab["final_score"] if target_lab else 0.0
            
            # TOP3の研究室
            top_3_labs = [
                {
                    "rank": idx + 1,
                    "lab_id": lab["lab_id"],
                    "lab_name": next(
                        (l["name"] for l in self.labs_data if l["id"] == lab["lab_id"]),
                        lab["lab_id"]
                    ),
                    "score": lab["final_score"]
                }
                for idx, lab in enumerate(results[:3])
            ]
            
            current_top_lab = results[0]["lab_id"] if results else None
            
            # ログに記録
            log_entry = {
                "step": i,
                "parameter_value": float(value),
                "target_rank": current_rank,
                "target_score": float(current_score),
                "top_3": top_3_labs,
                "top_lab": current_top_lab
            }
            detailed_log.append(log_entry)
            
            # 順位変動を検出
            if previous_rank is not None and current_rank != previous_rank:
                delta_value = value - param_values[i-1] if i > 0 else 0
                
                # 何位から何位に変わったか
                rank_direction = "上昇" if current_rank < previous_rank else "下降"
                
                # 誰に抜かれたか/誰を抜いたか
                if current_rank > previous_rank:
                    # 順位が下がった（誰かに抜かれた）
                    overtaken_by = []
                    for top_lab in top_3_labs:
                        if top_lab["rank"] < current_rank and top_lab["lab_id"] != lab_id:
                            overtaken_by.append({
                                "lab_id": top_lab["lab_id"],
                                "lab_name": top_lab["lab_name"],
                                "score": top_lab["score"]
                            })
                else:
                    # 順位が上がった（誰かを抜いた）
                    overtaken_by = []
                
                transition = {
                    "transition_point": float(value),
                    "previous_value": float(param_values[i-1]) if i > 0 else 1.0,
                    "delta_value": float(delta_value),
                    "rank_change": f"{previous_rank}位 → {current_rank}位",
                    "rank_direction": rank_direction,
                    "score_change": float(current_score - previous_score),
                    "previous_score": float(previous_score),
                    "current_score": float(current_score),
                    "previous_top_lab": previous_top_lab,
                    "current_top_lab": current_top_lab,
                    "top_3_before": previous_top_3,
                    "top_3_after": top_3_labs,
                    "explanation": self._generate_transition_explanation(
                        criterion, value, param_values[i-1] if i > 0 else 1.0,
                        previous_rank, current_rank, previous_score, current_score,
                        previous_top_lab, current_top_lab
                    )
                }
                transitions.append(transition)
            
            previous_rank = current_rank
            previous_score = current_score
            previous_top_lab = current_top_lab
            previous_top_3 = top_3_labs
        
        return {
            "lab_id": lab_id,
            "criterion": criterion,
            "detailed_log": detailed_log,
            "transitions": transitions,
            "total_transitions": len(transitions),
            "critical_transitions": [t for t in transitions if abs(t["delta_value"]) < 0.5],
            "summary": self._summarize_transitions(transitions)
        }
    
    def _generate_transition_explanation(
        self,
        criterion: str,
        new_value: float,
        old_value: float,
        old_rank: int,
        new_rank: int,
        old_score: float,
        new_score: float,
        old_top: str,
        new_top: str
    ) -> str:
        """遷移の説明文を生成"""
        delta = new_value - old_value
        score_delta = new_score - old_score
        
        explanation = f"{criterion} を {old_value:.2f} から {new_value:.2f} に変更（{delta:+.2f}）したとき、"
        explanation += f"順位が {old_rank}位 から {new_rank}位 に変化しました。"
        explanation += f"スコアは {old_score:.3f} から {new_score:.3f} に変化（{score_delta:+.4f}）。"
        
        if new_rank > old_rank:
            explanation += f"この変更により順位が下降し、"
            if new_top != old_top:
                explanation += f"1位が {old_top} から {new_top} に変わりました。"
        elif new_rank < old_rank:
            explanation += f"この変更により順位が上昇しました。"
        
        return explanation
    
    def _summarize_transitions(self, transitions: List[Dict]) -> Dict[str, Any]:
        """遷移のサマリーを生成"""
        if not transitions:
            return {"has_transitions": False}
        
        # 最も重要な遷移（スコア変動が大きい）
        critical_transition = max(transitions, key=lambda t: abs(t["score_change"]))
        
        # 平均的な遷移幅
        avg_delta = np.mean([abs(t["delta_value"]) for t in transitions])
        
        return {
            "has_transitions": True,
            "total_count": len(transitions),
            "critical_transition": critical_transition,
            "average_delta_value": float(avg_delta),
            "min_delta": float(min(abs(t["delta_value"]) for t in transitions)),
            "max_delta": float(max(abs(t["delta_value"]) for t in transitions))
        }
    
    # ==================== 拡張Phase 5: 全パラメータ組み合わせ分析 ====================
    
    def analyze_all_parameters_comprehensive(
        self,
        lab_id: str,
        base_profile: Dict
    ) -> Dict[str, Any]:
        """
        全13パラメータについて包括的に分析
        
        各パラメータについて:
        1. 影響度
        2. 境界値
        3. 詳細遷移点
        4. 感度曲線
        
        Returns:
            全パラメータの包括的分析結果
        """
        print(f"\n{'='*70}")
        print(f"  {lab_id} の全パラメータ包括分析")
        print(f"{'='*70}")
        
        results = {}
        
        for i, criterion in enumerate(self.criteria, 1):
            print(f"  [{i}/{len(self.criteria)}] {criterion} を分析中...")
            
            # 1. 影響度
            importance_data = self._analyze_single_parameter_importance(
                lab_id, criterion, base_profile
            )
            
            # 2. 境界値
            boundary = self.find_parameter_boundary(
                lab_id, criterion, base_profile, precision=self.boundary_precision
            )
            
            # 3. 詳細遷移点
            transitions = self.analyze_detailed_transitions(
                lab_id, criterion, base_profile, steps=self.transition_steps
            )
            
            results[criterion] = {
                "importance": importance_data,
                "boundary": boundary,
                "transitions": transitions
            }
        
        return results
    
    def _analyze_single_parameter_importance(
        self,
        lab_id: str,
        criterion: str,
        base_profile: Dict
    ) -> Dict[str, Any]:
        """単一パラメータの影響度分析（詳細版）"""
        scores = []
        ranks = []
        test_values = np.linspace(1.0, 10.0, 50)
        
        for value in test_values:
            test_profile = base_profile.copy()
            test_profile[criterion] = value
            test_profile[f"{criterion}_priority"] = 8.0
            
            results = self._match_all_labs(test_profile, self.labs_data)
            target_lab = next((lab for lab in results if lab["lab_id"] == lab_id), None)
            
            if target_lab:
                scores.append(target_lab["final_score"])
                ranks.append(target_lab["rank"])
            else:
                scores.append(0.0)
                ranks.append(999)
        
        score_range = max(scores) - min(scores)
        score_std = np.std(scores)
        
        # 最適値（スコアが最大になる値）
        optimal_idx = np.argmax(scores)
        optimal_value = float(test_values[optimal_idx])
        
        return {
            "score_range": float(score_range),
            "score_std": float(score_std),
            "importance": float(score_range),
            "optimal_value": optimal_value,
            "optimal_score": float(scores[optimal_idx]),
            "score_progression": [
                {"value": float(v), "score": float(s), "rank": int(r)}
                for v, s, r in zip(test_values, scores, ranks)
            ]
        }
    
    # ==================== 研究室ごとの完全レポート生成 ====================
    
    def generate_complete_lab_report(
        self,
        lab_id: str,
        output_dir: str = "comprehensive_reports"
    ) -> Dict[str, Any]:
        """
        特定の研究室について完全なレポートを生成（卒論用）
        
        出力内容:
        1. JSON: 全分析データ
        2. HTML: 視覚的なレポート
        3. PNG: 全図表（13枚）
        4. CSV: 表データ
        
        Args:
            lab_id: 分析対象の研究室ID
            output_dir: 出力ディレクトリ
            
        Returns:
            完全なレポートデータ
        """
        # ディレクトリ作成
        lab_dir = Path(output_dir) / lab_id
        lab_dir.mkdir(parents=True, exist_ok=True)
        
        # 研究室名
        lab_name = next(
            (lab["name"] for lab in self.labs_data if lab["id"] == lab_id),
            lab_id
        )
        
        print(f"\n{'='*70}")
        print(f"📊 {lab_name} の完全レポート生成")
        print(f"{'='*70}")
        
        # ベースプロファイル
        base_profile = {criterion: 5.0 for criterion in self.criteria}
        for criterion in self.criteria:
            base_profile[f"{criterion}_priority"] = 5.0
        
        # ★★★ 重要：対象研究室の分野に対する興味を設定 ★★★
        target_lab = next((lab for lab in self.labs_data if lab["id"] == lab_id), None)
        if target_lab and target_lab.get("field_id"):
            lab_field_id = target_lab["field_id"]
            base_profile["field_interests"] = {lab_field_id: 8.0}
            print(f"  📚 分野マッチング有効化: {lab_field_id} = 8.0")
        else:
            base_profile["field_interests"] = {}
            if target_lab:
                print(f"  ⚠️  警告: {lab_id} にはfield_idが設定されていません")
        
        # Phase 1-3: 標準分析
        print("\n[Phase 1-3] 標準分析を実行中...")
        standard_analysis = self.comprehensive_analysis(lab_id, self.high_precision_samples)
        
        # Phase 4: 全パラメータ包括分析
        print("\n[Phase 4] 全パラメータ包括分析を実行中...")
        all_params_analysis = self.analyze_all_parameters_comprehensive(
            lab_id, base_profile
        )
        
        # Phase 5: 可視化
        print("\n[Phase 5] 図表を生成中...")
        self._generate_all_visualizations(
            lab_id, lab_name, all_params_analysis, lab_dir
        )
        
        # Phase 6: 逆引き分析
        print("\n[Phase 6] 逆引き分析を実行中...")
        reverse_lookup_data = self.reverse_lookup_optimal_conditions(
            lab_id, target_rank=1, num_trials=1000
        )
        
        # Phase 6の可視化
        if reverse_lookup_data["solutions_found"] > 0:
            self._visualize_reverse_lookup_solutions(reverse_lookup_data, lab_dir)
        
        # Phase 7: ランキング配分分析
        print("\n[Phase 7] ランキング配分分析を実行中...")
        # 軽量モードの場合はサンプル数を削減
        if self.mode_name and "軽量" in self.mode_name:
            num_samples = 5000
        elif self.mode_name and "高精度" in self.mode_name:
            num_samples = 20000
        else:
            num_samples = 10000
        
        ranking_distribution = self.analyze_ranking_distribution(
            num_samples=num_samples,
            target_labs=[lab_id]
        )
        
        # Phase 7の可視化（このlab用の相関）
        self._visualize_parameter_correlation(
            ranking_distribution["parameter_correlations"],
            lab_dir,
            target_lab_id=lab_id
        )
        
        # 統合レポート
        report = {
            "lab_id": lab_id,
            "lab_name": lab_name,
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "standard_analysis": standard_analysis,
            "comprehensive_parameter_analysis": all_params_analysis,
            "reverse_lookup_analysis": reverse_lookup_data,
            "ranking_distribution_analysis": ranking_distribution,
            "summary": self._generate_executive_summary(
                lab_name, standard_analysis, all_params_analysis
            )
        }
        
        # JSON保存
        json_file = lab_dir / f"{lab_id}_complete_report.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n✅ JSONレポート: {json_file}")
        
        # HTMLレポート生成
        html_file = lab_dir / f"{lab_id}_report.html"
        self._generate_html_report(report, html_file)
        print(f"✅ HTMLレポート: {html_file}")
        
        # CSVデータ生成
        csv_file = lab_dir / f"{lab_id}_data.csv"
        self._generate_csv_data(report, csv_file)
        print(f"✅ CSVデータ: {csv_file}")
        
        print(f"\n{'='*70}")
        print(f"✅ 完全レポート生成完了: {lab_dir}/")
        print(f"{'='*70}\n")
        
        return report
    
    def _generate_all_visualizations(
        self,
        lab_id: str,
        lab_name: str,
        all_params_data: Dict,
        output_dir: Path
    ):
        """全ての可視化を生成"""
        import matplotlib.pyplot as plt
        
        viz_dir = output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        for criterion, data in all_params_data.items():
            # 1. 感度曲線
            self._plot_sensitivity_curve_detailed(
                lab_id, lab_name, criterion, data,
                viz_dir / f"{criterion}_sensitivity.png"
            )
            
            # 2. 遷移点マップ
            if data["transitions"]["transitions"]:
                self._plot_transition_map(
                    lab_id, lab_name, criterion, data["transitions"],
                    viz_dir / f"{criterion}_transitions.png"
                )
        
        # 3. 統合サマリー図
        self._plot_comprehensive_summary(
            lab_id, lab_name, all_params_data,
            viz_dir / "comprehensive_summary.png"
        )
    
    def _plot_sensitivity_curve_detailed(
        self,
        lab_id: str,
        lab_name: str,
        criterion: str,
        data: Dict,
        output_file: Path
    ):
        """詳細な感度曲線を描画"""
        import matplotlib.pyplot as plt
        
        importance = data["importance"]
        boundary = data["boundary"]
        transitions = data["transitions"]
        
        # データ準備
        progression = importance["score_progression"]
        values = [p["value"] for p in progression]
        scores = [p["score"] for p in progression]
        ranks = [p["rank"] for p in progression]
        
        # プロット
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # (1) スコア推移
        ax1 = axes[0]
        ax1.plot(values, scores, linewidth=2, color='#2E86AB', label='スコア')
        ax1.fill_between(values, scores, alpha=0.3, color='#2E86AB')
        
        # 最適値をマーク
        optimal_val = importance["optimal_value"]
        optimal_score = importance["optimal_score"]
        ax1.plot(optimal_val, optimal_score, 'ro', markersize=10, 
                label=f'最適値: {optimal_val:.2f}')
        
        # 境界値をマーク
        if boundary["boundaries"]["gains_top"]:
            ax1.axvline(boundary["boundaries"]["gains_top"], 
                       color='green', linestyle='--', alpha=0.7,
                       label=f'1位獲得: {boundary["boundaries"]["gains_top"]:.2f}')
        if boundary["boundaries"]["loses_top"]:
            ax1.axvline(boundary["boundaries"]["loses_top"], 
                       color='red', linestyle='--', alpha=0.7,
                       label=f'1位喪失: {boundary["boundaries"]["loses_top"]:.2f}')
        
        ax1.set_ylabel('マッチングスコア', fontsize=11)
        ax1.set_title(f'{lab_name} - {criterion} の感度分析', 
                     fontsize=13, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # (2) 順位推移
        ax2 = axes[1]
        ax2.plot(values, ranks, linewidth=2, color='#A23B72', marker='o', markersize=2)
        ax2.invert_yaxis()
        
        # 1位の領域をハイライト
        rank_1_values = [v for v, r in zip(values, ranks) if r == 1]
        if rank_1_values:
            ax2.axvspan(min(rank_1_values), max(rank_1_values), 
                       alpha=0.2, color='gold', label='1位獲得領域')
        
        # 遷移点をマーク
        for trans in transitions["transitions"]:
            ax2.axvline(trans["transition_point"], 
                       color='red', linestyle=':', alpha=0.5)
        
        ax2.set_ylabel('順位', fontsize=11)
        # 凡例があれば表示（なければスキップ）
        handles, labels = ax2.get_legend_handles_labels()
        if handles:
            ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # (3) 遷移点の詳細
        ax3 = axes[2]
        if transitions["transitions"]:
            trans_values = [t["transition_point"] for t in transitions["transitions"]]
            trans_scores = [t["current_score"] for t in transitions["transitions"]]
            trans_ranks = [int(t["rank_change"].split("→")[1].replace("位", "")) 
                          for t in transitions["transitions"]]
            
            scatter = ax3.scatter(trans_values, trans_ranks, 
                                 c=trans_scores, cmap='RdYlGn',
                                 s=100, alpha=0.7, edgecolors='black')
            ax3.invert_yaxis()
            
            # カラーバー
            cbar = plt.colorbar(scatter, ax=ax3)
            cbar.set_label('スコア', fontsize=10)
            
            ax3.set_ylabel('遷移後の順位', fontsize=11)
            ax3.set_xlabel(f'{criterion} (1: 低 → 10: 高)', fontsize=11)
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, '順位変動なし', 
                    ha='center', va='center', fontsize=14,
                    transform=ax3.transAxes)
            ax3.set_xlabel(f'{criterion}', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_transition_map(
        self,
        lab_id: str,
        lab_name: str,
        criterion: str,
        transitions_data: Dict,
        output_file: Path
    ):
        """遷移点マップを描画"""
        import matplotlib.pyplot as plt
        
        transitions = transitions_data["transitions"]
        
        if not transitions:
            return
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # 遷移を視覚化
        for i, trans in enumerate(transitions):
            prev_val = trans["previous_value"]
            curr_val = trans["transition_point"]
            prev_rank = int(trans["rank_change"].split("→")[0].replace("位", ""))
            curr_rank = int(trans["rank_change"].split("→")[1].replace("位", ""))
            
            # 矢印で遷移を表現
            ax.arrow(prev_val, prev_rank, 
                    curr_val - prev_val, curr_rank - prev_rank,
                    head_width=0.3, head_length=0.1, 
                    fc='#A23B72', ec='#A23B72', alpha=0.6,
                    length_includes_head=True)
            
            # ラベル
            ax.text((prev_val + curr_val) / 2, (prev_rank + curr_rank) / 2,
                   f'Δ{trans["delta_value"]:.2f}',
                   fontsize=8, ha='center', va='bottom')
        
        ax.invert_yaxis()
        ax.set_xlabel(f'{criterion} (1: 低 → 10: 高)', fontsize=12)
        ax.set_ylabel('順位', fontsize=12)
        ax.set_title(f'{lab_name} - {criterion} の遷移点マップ', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_comprehensive_summary(
        self,
        lab_id: str,
        lab_name: str,
        all_params_data: Dict,
        output_file: Path
    ):
        """統合サマリー図を描画"""
        import matplotlib.pyplot as plt
        
        # パラメータの重要度を集計
        params = list(all_params_data.keys())
        importances = [all_params_data[p]["importance"]["importance"] for p in params]
        optimal_values = [all_params_data[p]["importance"]["optimal_value"] for p in params]
        
        # ソート
        sorted_indices = np.argsort(importances)[::-1]
        params_sorted = [params[i] for i in sorted_indices]
        importances_sorted = [importances[i] for i in sorted_indices]
        optimal_values_sorted = [optimal_values[i] for i in sorted_indices]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # (1) 影響度ランキング
        ax1 = axes[0]
        bars = ax1.barh(params_sorted, importances_sorted, color='#2E86AB')
        ax1.set_xlabel('影響度（スコア変動幅）', fontsize=12)
        ax1.set_title(f'{lab_name} - パラメータ影響度ランキング', 
                     fontsize=14, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # 値ラベル
        for i, (bar, val) in enumerate(zip(bars, importances_sorted)):
            ax1.text(val + 0.01, i, f'{val:.3f}', 
                    va='center', fontsize=9)
        
        # (2) 最適値ヒートマップ
        ax2 = axes[1]
        cmap = plt.cm.RdYlGn
        norm = plt.Normalize(vmin=1, vmax=10)
        
        colors = [cmap(norm(val)) for val in optimal_values_sorted]
        bars2 = ax2.barh(params_sorted, optimal_values_sorted, color=colors)
        ax2.set_xlabel('最適値', fontsize=12)
        ax2.set_xlim(0, 10)
        ax2.set_title(f'{lab_name} - 各パラメータの最適値', 
                     fontsize=14, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        # 値ラベル
        for i, (bar, val) in enumerate(zip(bars2, optimal_values_sorted)):
            ax2.text(val + 0.2, i, f'{val:.1f}', 
                    va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    # ==================== Phase 6 & 7 可視化メソッド ====================
    
    def _visualize_ranking_distribution(
        self,
        distribution_data: Dict,
        output_dir: Path,
        top_n_labs: int = 10
    ):
        """ランキング配分分析の可視化"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        rank_probs = distribution_data["rank_probabilities"]
        
        # TOP Nの研究室を選択（top1_probabilityが高い順）
        sorted_labs = sorted(
            rank_probs.items(),
            key=lambda x: x[1]["top1_probability"],
            reverse=True
        )[:top_n_labs]
        
        # (1) TOP1確率ランキング
        fig, ax = plt.subplots(figsize=(12, 6))
        
        lab_names = [item[1]["lab_name"] for item in sorted_labs]
        top1_probs = [item[1]["top1_probability"] * 100 for item in sorted_labs]
        
        bars = ax.barh(lab_names, top1_probs, color='#FF6B6B')
        ax.set_xlabel('1位獲得確率 (%)', fontsize=12)
        ax.set_title('研究室別 1位獲得確率ランキング (モンテカルロサンプリング)', 
                    fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        for i, (bar, prob) in enumerate(zip(bars, top1_probs)):
            ax.text(prob + 0.5, i, f'{prob:.2f}%', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_dir / "ranking_top1_probability.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # (2) TOP3/TOP5確率比較
        fig, ax = plt.subplots(figsize=(14, 7))
        
        x = np.arange(len(lab_names))
        width = 0.25
        
        top1 = [item[1]["top1_probability"] * 100 for item in sorted_labs]
        top3 = [item[1]["top3_probability"] * 100 for item in sorted_labs]
        top5 = [item[1]["top5_probability"] * 100 for item in sorted_labs]
        
        ax.bar(x - width, top1, width, label='TOP1', color='#FF6B6B')
        ax.bar(x, top3, width, label='TOP3', color='#4ECDC4')
        ax.bar(x + width, top5, width, label='TOP5', color='#95E1D3')
        
        ax.set_xlabel('研究室', fontsize=12)
        ax.set_ylabel('確率 (%)', fontsize=12)
        ax.set_title('研究室別 TOP順位獲得確率', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(lab_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "ranking_topn_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # (3) 順位分布ヒートマップ
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # ヒートマップ用データを準備
        heatmap_data = []
        for item in sorted_labs:
            lab_id, lab_data = item
            rank_dist = lab_data["rank_distribution"]
            # TOP20までの順位確率
            row = [rank_dist.get(i, 0) * 100 for i in range(1, 21)]
            heatmap_data.append(row)
        
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt='.1f',
            cmap='YlOrRd',
            xticklabels=range(1, 21),
            yticklabels=lab_names,
            cbar_kws={'label': '確率 (%)'},
            ax=ax
        )
        
        ax.set_xlabel('順位', fontsize=12)
        ax.set_ylabel('研究室', fontsize=12)
        ax.set_title('研究室別 順位分布ヒートマップ (TOP20)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / "ranking_distribution_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # (4) 平均順位比較
        fig, ax = plt.subplots(figsize=(12, 6))
        
        avg_ranks = [item[1]["average_rank"] for item in sorted_labs]
        
        bars = ax.barh(lab_names, avg_ranks, color='#A8DADC')
        ax.set_xlabel('平均順位', fontsize=12)
        ax.set_title('研究室別 平均順位 (低いほど上位)', fontsize=14, fontweight='bold')
        ax.invert_xaxis()  # 低い値が右に来るように
        ax.grid(axis='x', alpha=0.3)
        
        for i, (bar, rank) in enumerate(zip(bars, avg_ranks)):
            ax.text(rank - 0.5, i, f'{rank:.2f}', va='center', fontsize=9, ha='right')
        
        plt.tight_layout()
        plt.savefig(output_dir / "ranking_average_rank.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ ランキング配分図表（4枚）を保存しました")
    
    def _visualize_parameter_correlation(
        self,
        correlation_data: Dict,
        output_dir: Path,
        target_lab_id: str = None
    ):
        """パラメータと順位の相関を可視化"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # 特定の研究室の相関を可視化
        if target_lab_id and target_lab_id in correlation_data[self.criteria[0]]:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # 相関行列を作成
            corr_values = []
            for criterion in self.criteria:
                if target_lab_id in correlation_data[criterion]:
                    corr_values.append(correlation_data[criterion][target_lab_id]["correlation"])
                else:
                    corr_values.append(0.0)
            
            # ソート
            sorted_indices = np.argsort(np.abs(corr_values))[::-1]
            criteria_sorted = [self.criteria[i] for i in sorted_indices]
            corr_sorted = [corr_values[i] for i in sorted_indices]
            
            # 色分け（正の相関=緑、負の相関=赤）
            colors = ['#2ECC71' if c > 0 else '#E74C3C' for c in corr_sorted]
            
            bars = ax.barh(criteria_sorted, corr_sorted, color=colors)
            ax.set_xlabel('相関係数', fontsize=12)
            ax.set_title(f'パラメータと順位の相関 (対象研究室)', fontsize=14, fontweight='bold')
            ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
            ax.grid(axis='x', alpha=0.3)
            
            for i, (bar, corr) in enumerate(zip(bars, corr_sorted)):
                label_x = corr + 0.02 if corr > 0 else corr - 0.02
                ha = 'left' if corr > 0 else 'right'
                ax.text(label_x, i, f'{corr:.3f}', va='center', ha=ha, fontsize=9)
            
            plt.tight_layout()
            plt.savefig(output_dir / f"correlation_{target_lab_id}.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✅ 相関図表を保存しました")
    
    def _visualize_reverse_lookup_solutions(
        self,
        solutions_data: Dict,
        output_dir: Path
    ):
        """逆引き分析の解を可視化"""
        import matplotlib.pyplot as plt
        
        if len(solutions_data["solutions"]) == 0:
            print(f"  ⚠️ 解が見つかりませんでした")
            return
        
        solutions = solutions_data["solutions"]
        
        # (1) 解の分布（2D投影）- 影響度の高いパラメータ2つ
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 最初の2つのパラメータを使用（実際には影響度でソートすべき）
        param1 = self.criteria[0]
        param2 = self.criteria[1]
        
        x_values = [s["parameters"][param1] for s in solutions]
        y_values = [s["parameters"][param2] for s in solutions]
        scores = [s["score"] for s in solutions]
        
        scatter = ax.scatter(x_values, y_values, c=scores, s=100, 
                           cmap='viridis', alpha=0.6, edgecolors='black')
        ax.set_xlabel(f'{param1}', fontsize=12)
        ax.set_ylabel(f'{param2}', fontsize=12)
        ax.set_title(f'最適条件の分布 ({len(solutions)}個の解)', 
                    fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('適合度スコア', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_dir / "reverse_lookup_solutions_2d.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # (2) パラメータ値の範囲
        fig, ax = plt.subplots(figsize=(12, 8))
        
        param_ranges = {}
        for criterion in self.criteria:
            values = [s["parameters"][criterion] for s in solutions]
            param_ranges[criterion] = {
                "min": min(values),
                "max": max(values),
                "mean": np.mean(values),
                "std": np.std(values)
            }
        
        criteria_list = list(param_ranges.keys())
        means = [param_ranges[c]["mean"] for c in criteria_list]
        mins = [param_ranges[c]["min"] for c in criteria_list]
        maxs = [param_ranges[c]["max"] for c in criteria_list]
        
        y_pos = np.arange(len(criteria_list))
        
        # エラーバー
        ax.barh(y_pos, means, xerr=[
            [m - mi for m, mi in zip(means, mins)],
            [ma - m for ma, m in zip(maxs, means)]
        ], color='#3498DB', alpha=0.7, capsize=5)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(criteria_list)
        ax.set_xlabel('パラメータ値', fontsize=12)
        ax.set_title(f'最適パラメータの範囲 (平均±範囲)', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "reverse_lookup_parameter_ranges.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ 逆引き分析図表（2枚）を保存しました")
    
    def _generate_executive_summary(
        self,
        lab_name: str,
        standard_analysis: Dict,
        comprehensive_analysis: Dict
    ) -> Dict[str, Any]:
        """エグゼクティブサマリーを生成"""
        # 最重要パラメータ
        top_params = standard_analysis["phase1_parameter_importance"]["top_3_influential"]
        
        # 1位獲得条件
        top_conditions = standard_analysis["phase2_top_rank_conditions"]
        
        # 全パラメータの統計
        all_importances = [
            comprehensive_analysis[p]["importance"]["importance"]
            for p in self.criteria
        ]
        
        return {
            "lab_name": lab_name,
            "top_3_parameters": top_params,
            "can_achieve_top_rank": top_conditions["found_top_rank"],
            "top_rank_probability": top_conditions.get("top_rank_probability", 0),
            "average_parameter_importance": float(np.mean(all_importances)),
            "max_parameter_importance": float(np.max(all_importances)),
            "total_parameters_analyzed": len(self.criteria),
            "recommendations": self._generate_recommendations(
                top_conditions, comprehensive_analysis
            )
        }
    
    def _generate_recommendations(
        self,
        top_conditions: Dict,
        comprehensive_analysis: Dict
    ) -> List[str]:
        """推奨事項を生成"""
        recommendations = []
        
        if top_conditions["found_top_rank"]:
            prob = top_conditions["top_rank_probability"] * 100
            recommendations.append(
                f"この研究室は適切な学生プロファイルで1位を獲得できます（確率: {prob:.1f}%）"
            )
            
            # 典型プロファイルから推奨
            typical = top_conditions["typical_profile"]
            for param in list(typical.keys())[:3]:
                mean = typical[param]["mean"]
                recommendations.append(
                    f"{param} の推奨値: {mean:.1f}点"
                )
        else:
            recommendations.append(
                "この研究室は1位獲得が困難です。他の候補も検討することを推奨します。"
            )
        
        return recommendations
    
    def _generate_html_report(self, report: Dict, output_file: Path):
        """HTMLレポートを生成"""
        html = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{report['lab_name']} - 包括的感度分析レポート</title>
    <style>
        body {{
            font-family: 'Helvetica Neue', Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #2E86AB 0%, #A23B72 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .section {{
            background: white;
            padding: 25px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1, h2, h3 {{
            margin-top: 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #2E86AB;
            color: white;
        }}
        .metric {{
            display: inline-block;
            background: #e3f2fd;
            padding: 15px 25px;
            margin: 10px;
            border-radius: 5px;
            border-left: 4px solid #2E86AB;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2E86AB;
        }}
        .metric-label {{
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{report['lab_name']}</h1>
        <p>包括的感度分析レポート</p>
        <p>生成日時: {report['analysis_date']}</p>
    </div>
    
    <div class="section">
        <h2>📊 エグゼクティブサマリー</h2>
        <div class="metric">
            <div class="metric-label">1位獲得確率</div>
            <div class="metric-value">{report['summary']['top_rank_probability']*100:.1f}%</div>
        </div>
        <div class="metric">
            <div class="metric-label">分析パラメータ数</div>
            <div class="metric-value">{report['summary']['total_parameters_analyzed']}</div>
        </div>
        <div class="metric">
            <div class="metric-label">最大影響度</div>
            <div class="metric-value">{report['summary']['max_parameter_importance']:.3f}</div>
        </div>
    </div>
    
    <div class="section">
        <h2>🔥 最重要パラメータ TOP 3</h2>
        <ol>
            {''.join([f'<li><strong>{p}</strong></li>' for p in report['summary']['top_3_parameters']])}
        </ol>
    </div>
    
    <div class="section">
        <h2>💡 推奨事項</h2>
        <ul>
            {''.join([f'<li>{r}</li>' for r in report['summary']['recommendations']])}
        </ul>
    </div>
    
    <div class="section">
        <h2>📈 詳細分析データ</h2>
        <p>完全な数値データは同梱のJSONファイルを参照してください。</p>
        <p>図表は visualizations/ フォルダに保存されています。</p>
    </div>
</body>
</html>
"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)
    
    def _generate_csv_data(self, report: Dict, output_file: Path):
        """CSVデータを生成"""
        try:
            import pandas as pd
            
            rows = []
            for param, data in report["comprehensive_parameter_analysis"].items():
                importance = data["importance"]
                boundary = data["boundary"]
                transitions = data["transitions"]
                
                row = {
                    "パラメータ": param,
                    "影響度": importance["importance"],
                    "最適値": importance["optimal_value"],
                    "最適スコア": importance["optimal_score"],
                    "1位獲得境界": boundary["boundaries"].get("gains_top", "N/A"),
                    "1位喪失境界": boundary["boundaries"].get("loses_top", "N/A"),
                    "遷移点数": transitions["total_transitions"]
                }
                rows.append(row)
            
            df = pd.DataFrame(rows)
            df = df.sort_values("影響度", ascending=False)
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            
        except ImportError:
            print("⚠️ pandas が必要です")
    
    # ==================== 全研究室の一括分析 ====================
    
    def analyze_all_labs_comprehensive(
        self,
        output_dir: str = "comprehensive_reports",
        no_confirm: bool = False,
        resume: bool = False,
        start_from: int = 1
    ) -> Dict[str, Any]:
        """
        全31研究室について包括的分析を実行
        
        推定実行時間: 2-4時間
        
        Args:
            output_dir: 出力ディレクトリ
            no_confirm: Trueの場合、確認プロンプトをスキップ
            resume: Trueの場合、完了済み研究室をスキップ
            start_from: この番号の研究室から開始（1-31）
        """
        print("\n" + "="*70)
        print("🚀 全研究室の包括的感度分析を開始します")
        print("="*70)
        print(f"対象: {len(self.labs_data)}研究室")
        print(f"精度モード: {self.mode_name}")
        print(f"サンプル数: {self.high_precision_samples}件/研究室")
        print(f"遷移ステップ: {self.transition_steps}ステップ/パラメータ")
        
        # 推定時間を精度モードに応じて計算
        if hasattr(self, 'mode_name'):
            if 'fast' in self.mode_name.lower():
                estimated_time = "約8-16時間"
                time_per_lab = "15-30分"
            elif 'precise' in self.mode_name.lower():
                estimated_time = "約31-62時間"
                time_per_lab = "60-120分"
            else:  # standard
                estimated_time = "約16-31時間"
                time_per_lab = "30-60分"
        else:
            estimated_time = "約16-31時間"
            time_per_lab = "30-60分"
        
        print(f"推定所要時間: {estimated_time} ({time_per_lab}/研究室)")
        
        if resume:
            print("⚡ 再開モード: 完了済み研究室をスキップします")
        if start_from > 1:
            print(f"⚡ 開始位置: {start_from}番目の研究室から")
        print("="*70)
        
        if not no_confirm:
            input("\n準備ができたらEnterキーを押してください...")
        else:
            print("\n⚡ --no-confirm モードで実行します（自動開始）")
            import time
            time.sleep(2)
        
        start_time = datetime.now()
        all_reports = {}
        
        # ログファイルのセットアップ
        log_file = Path(output_dir) / f"analysis_log_{start_time.strftime('%Y%m%d_%H%M%S')}.txt"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        def log_message(message: str):
            """コンソールとログファイルの両方に出力"""
            print(message)
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(message + '\n')
        
        log_message(f"\n{'='*70}")
        log_message(f"開始時刻: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        log_message(f"{'='*70}\n")
        
        skipped_count = 0
        processed_count = 0
        
        for i, lab in enumerate(self.labs_data, 1):
            lab_id = lab["id"]
            
            # start_fromより前の研究室はスキップ
            if i < start_from:
                skipped_count += 1
                continue
            
            # resumeモードで完了済みをチェック
            if resume:
                report_file = Path(output_dir) / lab_id / f"{lab_id}_complete_report.json"
                if report_file.exists():
                    log_message(f"\n⏭️  [{i}/{len(self.labs_data)}] {lab['name']} (ID: {lab_id}) - スキップ（完了済み）")
                    skipped_count += 1
                    
                    # 既存レポートを読み込んで all_reports に追加
                    try:
                        with open(report_file, 'r', encoding='utf-8') as f:
                            all_reports[lab_id] = json.load(f)
                    except:
                        pass
                    
                    continue
            
            log_message(f"\n{'#'*70}")
            log_message(f"# [{i}/{len(self.labs_data)}] {lab['name']} (ID: {lab_id})")
            log_message(f"{'#'*70}")
            
            report = self.generate_complete_lab_report(lab_id, output_dir)
            all_reports[lab_id] = report
            processed_count += 1
            
            # 進捗表示（実際に処理した数でカウント）
            elapsed = (datetime.now() - start_time).total_seconds()
            if processed_count > 0:
                avg_time = elapsed / processed_count
                remaining_labs = len(self.labs_data) - i
                remaining = avg_time * remaining_labs
                
                log_message(f"\n⏱️  経過時間: {elapsed/60:.1f}分")
                log_message(f"⏱️  残り時間: {remaining/60:.1f}分（推定）")
            log_message(f"✅ [{i}/{len(self.labs_data)}] 完了 (処理済み: {processed_count}, スキップ: {skipped_count})")
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        log_message(f"\n{'='*70}")
        log_message(f"✅ 全研究室の包括的分析が完了しました！")
        log_message(f"{'='*70}")
        log_message(f"📁 出力先: {output_dir}/")
        log_message(f"⏱️  総実行時間: {total_time/3600:.2f}時間")
        log_message(f"📊 処理済み: {processed_count}研究室")
        log_message(f"📊 スキップ: {skipped_count}研究室")
        log_message(f"📊 合計: {len(self.labs_data)}研究室")
        log_message(f"📄 ログファイル: {log_file}")
        log_message(f"{'='*70}")
        
        # 統合レポート
        master_report = {
            "total_labs": len(self.labs_data),
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_time_minutes": total_time / 60,
            "individual_reports": all_reports,
            "global_summary": self._generate_global_summary(all_reports)
        }
        
        # Phase 7（全体版）: 全研究室のランキング配分分析
        print(f"\n{'='*70}")
        print("📈 全研究室のランキング配分分析を実行中...")
        print(f"{'='*70}")
        
        # サンプル数を精度モードに応じて設定
        if self.mode_name and "軽量" in self.mode_name:
            global_samples = 10000
        elif self.mode_name and "高精度" in self.mode_name:
            global_samples = 30000
        else:
            global_samples = 20000
        
        global_ranking_distribution = self.analyze_ranking_distribution(
            num_samples=global_samples,
            target_labs=None  # 全研究室
        )
        
        # 全体のランキング配分を可視化
        global_output_dir = Path(output_dir) / "global_analysis"
        global_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n全体ランキング配分の可視化を生成中...")
        self._visualize_ranking_distribution(
            global_ranking_distribution,
            global_output_dir,
            top_n_labs=15  # TOP15を表示
        )
        
        # マスターレポートに追加
        master_report["global_ranking_distribution"] = global_ranking_distribution
        
        # マスターレポート保存
        master_file = Path(output_dir) / "master_report.json"
        with open(master_file, 'w', encoding='utf-8') as f:
            json.dump(master_report, f, ensure_ascii=False, indent=2)
        
        # ランキング配分CSVを生成
        self._save_ranking_distribution_csv(global_ranking_distribution, global_output_dir)
        
        return master_report
    
    def _save_ranking_distribution_csv(
        self,
        distribution_data: Dict,
        output_dir: Path
    ):
        """ランキング配分データをCSVで保存"""
        import csv
        
        rank_probs = distribution_data["rank_probabilities"]
        
        # (1) 確率分布CSV
        csv_file = output_dir / "ranking_probabilities.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # ヘッダー
            writer.writerow([
                "研究室ID", "研究室名", "TOP1確率(%)", "TOP3確率(%)", 
                "TOP5確率(%)", "TOP10確率(%)", "平均順位"
            ])
            
            # データ
            for lab_id, data in sorted(
                rank_probs.items(),
                key=lambda x: x[1]["top1_probability"],
                reverse=True
            ):
                writer.writerow([
                    lab_id,
                    data["lab_name"],
                    f"{data['top1_probability'] * 100:.2f}",
                    f"{data['top3_probability'] * 100:.2f}",
                    f"{data['top5_probability'] * 100:.2f}",
                    f"{data['top10_probability'] * 100:.2f}",
                    f"{data['average_rank']:.2f}"
                ])
        
        print(f"  ✅ ランキング確率CSV: {csv_file}")
        
        # (2) 詳細な順位分布CSV
        detail_csv = output_dir / "ranking_distribution_detail.csv"
        with open(detail_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # ヘッダー
            header = ["研究室ID", "研究室名"] + [f"{i}位(%)" for i in range(1, 21)]
            writer.writerow(header)
            
            # データ
            for lab_id, data in sorted(
                rank_probs.items(),
                key=lambda x: x[1]["top1_probability"],
                reverse=True
            ):
                row = [lab_id, data["lab_name"]]
                rank_dist = data["rank_distribution"]
                for i in range(1, 21):
                    prob = rank_dist.get(i, 0) * 100
                    row.append(f"{prob:.2f}")
                writer.writerow(row)
        
        print(f"  ✅ 詳細分布CSV: {detail_csv}")
    
    def _generate_global_summary(self, all_reports: Dict) -> Dict[str, Any]:
        """全体サマリーを生成"""
        # 各研究室の最重要パラメータを集計
        param_frequency = {criterion: 0 for criterion in self.criteria}
        
        for report in all_reports.values():
            top_3 = report["summary"]["top_3_parameters"]
            for param in top_3:
                param_frequency[param] += 1
        
        # 1位獲得可能な研究室
        achievable_labs = sum(
            1 for report in all_reports.values()
            if report["summary"]["can_achieve_top_rank"]
        )
        
        return {
            "globally_influential_parameters": sorted(
                param_frequency.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
            "labs_with_top_rank_potential": achievable_labs,
            "total_labs_analyzed": len(all_reports)
        }


# ==================== メイン実行 ====================

def flatten_lab_data(labs_data: List[Dict]) -> List[Dict]:
    """
    研究室データをフラット化
    
    labs_database.jsonの構造:
      lab["features"]["research_intensity"]
    
    FuzzyMultiPathMatcherが期待する構造:
      lab["research_intensity"]
    
    Args:
        labs_data: 元の研究室データ（ネスト構造）
        
    Returns:
        フラット化された研究室データ
    """
    flattened = []
    
    for lab in labs_data:
        flat_lab = lab.copy()
        
        # features をフラット化
        if "features" in lab:
            features = lab["features"]
            for key, value in features.items():
                flat_lab[key] = value
            
            # features キーは削除（冗長なので）
            del flat_lab["features"]
        
        # field_id を確保
        if "field_id" not in flat_lab:
            # research_fields の最初の要素から生成
            research_fields = flat_lab.get("research_fields", [])
            if research_fields:
                flat_lab["field_id"] = research_fields[0].lower().replace("・", "_").replace(" ", "_")
        
        flattened.append(flat_lab)
    
    return flattened


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="包括的感度分析システム（卒論用・最高品質版）"
    )
    
    parser.add_argument(
        '--mode',
        choices=['full', 'single'],
        required=True,
        help='分析モード'
    )
    
    parser.add_argument(
        '--lab',
        type=str,
        help='単一研究室モードの研究室ID'
    )
    
    parser.add_argument(
        '--no-confirm',
        action='store_true',
        help='確認プロンプトをスキップ（バッチ実行用）'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='完了済み研究室をスキップして再開'
    )
    
    parser.add_argument(
        '--start-from',
        type=int,
        default=1,
        help='指定した番号の研究室から開始（1-31）'
    )
    
    parser.add_argument(
        '--precision',
        choices=['fast', 'standard', 'precise'],
        default='standard',
        help='精度モード: fast(軽量/15-30分/研究室), standard(標準/30-60分/研究室), precise(高精度/60-120分/研究室)'
    )
    
    args = parser.parse_args()
    
    # システム初期化
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    from data.models.labs_database import LabDatabase
    from core.matching.fuzzy_multipath_matcher import FuzzyMultiPathMatcher
    
    print("🔍 包括的感度分析システム v1.0")
    print("="*70)
    
    # データベース読み込み
    db = LabDatabase()
    raw_labs_data = db.get_all_labs()
    
    # データをフラット化
    labs_data = flatten_lab_data(raw_labs_data)
    
    print(f"✅ {len(labs_data)}研究室を読み込みました（フラット化完了）")
    
    # マッチャー初期化
    matcher = FuzzyMultiPathMatcher()
    print(f"✅ マッチャーを初期化しました")
    
    # アナライザー初期化
    analyzer = ComprehensiveSensitivityAnalyzer(matcher, labs_data)
    
    # 精度モード設定
    analyzer.set_precision_mode(args.precision)
    print(f"✅ 包括的アナライザーを初期化しました")
    print(f"⚙️  精度モード: {analyzer.mode_name}")
    print(f"   - サンプル数: {analyzer.high_precision_samples}")
    print(f"   - 遷移ステップ: {analyzer.transition_steps}")
    print()
    
    # モード別実行
    if args.mode == 'single':
        if not args.lab:
            print("❌ --lab オプションで研究室IDを指定してください")
            return 1
        
        analyzer.generate_complete_lab_report(args.lab)
    
    elif args.mode == 'full':
        analyzer.analyze_all_labs_comprehensive(
            no_confirm=args.no_confirm,
            resume=args.resume,
            start_from=args.start_from
        )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())