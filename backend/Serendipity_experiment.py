#!/usr/bin/env python3
"""
セレンディピティ評価実験
=======================
研究室配属推薦システムのセレンディピティ性を検証する実験スクリプト

実験内容:
- Phase 1: 仮想学生プロファイル生成（1000件）
- Phase 2: Primitive Model実装（PM1, PM2, PM3）
- Phase 3: 比較実験（PM vs 提案システム）
- Phase 4: セレンディピティ指標算出

理論的背景:
- Ge et al. (2010): Serendipity = Unexpected ∩ Useful
- Murakami et al. (2008): Primitive Model概念
- Kotkov et al. (2016): セレンディピティ = 関連性 × 新規性 × 意外性
"""

import json
import random
import numpy as np
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional
import math

# =============================================================================
# 定数定義
# =============================================================================

# 11項目の基本パラメータ
PARAMETERS = [
    'research_intensity',      # 研究強度
    'advisor_style',           # 指導スタイル
    'team_work',               # チームワーク
    'workload',                # ワークロード
    'theory_practice',         # 理論/実践バランス
    'lab_atmosphere',          # 研究室雰囲気
    'flexibility',             # 柔軟性
    'skill_development',       # スキル開発
    'publication_opportunity', # 発表機会
    'communication',           # コミュニケーション
    'interdisciplinary'        # 学際性
]

# 研究分野（31ゼミの分野）
RESEARCH_FIELDS = [
    'ai_ml',              # AI・機械学習
    'web_design_uiux',    # Webデザイン・UI/UX
    'game_dev',           # ゲーム開発
    'computer_music',     # コンピュータ音楽
    'audio_processing',   # 音響処理
    'illustration_art',   # イラスト・アート
    'media_art',          # メディアアート
    'graphic_visual',     # グラフィック・ビジュアル
    'video_production',   # 映像制作
    'animation',          # アニメーション
    'programming',        # プログラミング
    'data_science',       # データサイエンス
    'iot_embedded',       # IoT・組み込み
    'vr_ar',              # VR/AR
    'network_security'    # ネットワーク・セキュリティ
]

# 31ゼミのデータ（実際のデータに基づく）
LABS = {
    'saito_kenji': {'name': '齋藤健司ゼミ', 'field': 'ai_ml', 'popularity': 0.95},
    'ito_masahiko': {'name': '伊藤正彦ゼミ', 'field': 'programming', 'popularity': 0.85},
    'yumura': {'name': '湯村ゼミ', 'field': 'data_science', 'popularity': 0.80},
    'miura': {'name': '三浦ゼミ', 'field': 'network_security', 'popularity': 0.75},
    'matsui': {'name': '松井ゼミ', 'field': 'iot_embedded', 'popularity': 0.70},
    'shimada': {'name': '島田ゼミ', 'field': 'vr_ar', 'popularity': 0.72},
    'iijima': {'name': '飯嶋ゼミ', 'field': 'programming', 'popularity': 0.68},
    'sugisawa': {'name': '杉澤ゼミ', 'field': 'graphic_visual', 'popularity': 0.78},
    'morikawa': {'name': '森川ゼミ', 'field': 'video_production', 'popularity': 0.65},
    'taniguchi': {'name': '谷口ゼミ', 'field': 'animation', 'popularity': 0.70},
    'saito_hajime': {'name': '斎藤一ゼミ', 'field': 'programming', 'popularity': 0.60},
    'tsuji': {'name': '辻ゼミ', 'field': 'data_science', 'popularity': 0.62},
    'yasuda': {'name': '安田ゼミ', 'field': 'web_design_uiux', 'popularity': 0.75},
    'niiyama': {'name': '新井山ゼミ', 'field': 'iot_embedded', 'popularity': 0.55},
    'chikasawa': {'name': '近澤ゼミ', 'field': 'web_design_uiux', 'popularity': 0.80},
    'mukaida': {'name': '向田ゼミ', 'field': 'video_production', 'popularity': 0.58},
    'yamakita': {'name': '山北ゼミ', 'field': 'programming', 'popularity': 0.52},
    'hirooku': {'name': '広奥ゼミ', 'field': 'game_dev', 'popularity': 0.82},
    'hayata': {'name': '隼田ゼミ', 'field': 'media_art', 'popularity': 0.88},
    'sasaki': {'name': '佐々木ゼミ', 'field': 'graphic_visual', 'popularity': 0.60},
    'oshima': {'name': '大島ゼミ', 'field': 'animation', 'popularity': 0.65},
    'ureshino': {'name': '畝喜本ゼミ', 'field': 'illustration_art', 'popularity': 0.70},
    'sakamoto': {'name': '坂本ゼミ', 'field': 'computer_music', 'popularity': 0.55},
    'ito_marty': {'name': '伊藤マーティゼミ', 'field': 'illustration_art', 'popularity': 0.85},
    'kakinami': {'name': '柿並ゼミ', 'field': 'media_art', 'popularity': 0.62},
    'kawahara': {'name': '河原ゼミ', 'field': 'game_dev', 'popularity': 0.90},
    'kim': {'name': '金ゼミ', 'field': 'audio_processing', 'popularity': 0.50},
    'wataya': {'name': '綿谷ゼミ', 'field': 'vr_ar', 'popularity': 0.58},
    'mori': {'name': '守ゼミ', 'field': 'audio_processing', 'popularity': 0.52},
    'hirayama': {'name': '平山ゼミ', 'field': 'computer_music', 'popularity': 0.60},
    'fujiwara': {'name': '藤原ゼミ', 'field': 'network_security', 'popularity': 0.48}
}

# 各ゼミの11項目属性値（1-10スケール、仮想データ）
LAB_ATTRIBUTES = {}
random.seed(42)  # 再現性のため
for lab_id in LABS:
    LAB_ATTRIBUTES[lab_id] = {
        param: random.randint(3, 9) for param in PARAMETERS
    }

# =============================================================================
# データクラス定義
# =============================================================================

@dataclass
class StudentProfile:
    """学生プロファイル"""
    id: str
    preferences: Dict[str, int]      # 11項目の選好値（1-10）
    importance: Dict[str, int]       # 11項目の重要度（1-10）
    field_interests: List[str]       # 興味のある研究分野（1-3個）
    primary_field: str               # 第一希望の分野

@dataclass 
class RecommendationResult:
    """推薦結果"""
    lab_id: str
    lab_name: str
    total_score: float
    field_score: float
    environment_score: float
    rank: int

@dataclass
class SerendipityMetrics:
    """セレンディピティ評価指標"""
    serendipity_score: float         # セレンディピティスコア
    unexpectedness: float            # 意外性
    relevance: float                 # 関連性
    is_serendipitous: bool           # セレンディピティ的かどうか

# =============================================================================
# Phase 1: 仮想学生プロファイル生成
# =============================================================================

def generate_student_profiles(n: int = 1000, seed: int = 42) -> List[StudentProfile]:
    """
    仮想学生プロファイルを生成
    
    Parameters:
        n: 生成するプロファイル数
        seed: 乱数シード（再現性のため）
    
    Returns:
        StudentProfileのリスト
    """
    random.seed(seed)
    np.random.seed(seed)
    
    profiles = []
    
    for i in range(n):
        # 11項目の選好値を生成（正規分布、平均5.5、標準偏差2）
        preferences = {}
        for param in PARAMETERS:
            value = int(np.clip(np.random.normal(5.5, 2), 1, 10))
            preferences[param] = value
        
        # 重要度を生成（一様分布）
        importance = {}
        for param in PARAMETERS:
            importance[param] = random.randint(1, 10)
        
        # 興味分野を1-3個選択
        num_interests = random.randint(1, 3)
        field_interests = random.sample(RESEARCH_FIELDS, num_interests)
        primary_field = field_interests[0]
        
        profile = StudentProfile(
            id=f"student_{i+1:04d}",
            preferences=preferences,
            importance=importance,
            field_interests=field_interests,
            primary_field=primary_field
        )
        profiles.append(profile)
    
    return profiles

# =============================================================================
# 提案システム（ファジィ決定木ベース）の簡易実装
# =============================================================================

def fuzzy_membership(value: float, target: float, sigma: float = 0.2) -> float:
    """
    ファジィメンバーシップ関数（ガウス型）
    
    Parameters:
        value: 学生の選好値（正規化済み）
        target: ゼミの属性値（正規化済み）
        sigma: 許容幅
    
    Returns:
        メンバーシップ度（0-1）
    """
    diff = abs(value - target)
    return math.exp(-(diff ** 2) / (2 * sigma ** 2))

def calculate_field_match(student_fields: List[str], lab_field: str) -> float:
    """
    研究分野のマッチング度を計算
    
    Parameters:
        student_fields: 学生の興味分野リスト
        lab_field: ゼミの研究分野
    
    Returns:
        マッチング度（0.3-1.0）
    """
    if lab_field in student_fields:
        # 第一希望なら1.0、それ以外なら位置に応じて減衰
        try:
            position = student_fields.index(lab_field)
            return 1.0 - (position * 0.15)  # 0.85, 0.70 for 2nd, 3rd
        except ValueError:
            return 0.3  # マッチしない場合
    return 0.3  # 最低値

def proposed_system_recommend(profile: StudentProfile) -> List[RecommendationResult]:
    """
    提案システム（ファジィ決定木）による推薦
    
    重み付きファジィマッチングスコアを計算し、研究分野マッチを統合
    
    Parameters:
        profile: 学生プロファイル
    
    Returns:
        推薦結果のリスト（スコア降順）
    """
    results = []
    
    for lab_id, lab_info in LABS.items():
        lab_attrs = LAB_ATTRIBUTES[lab_id]
        
        # 環境適合度スコア（11項目の重み付きファジィマッチング）
        weighted_sum = 0
        weight_total = 0
        
        for param in PARAMETERS:
            student_pref = profile.preferences[param] / 10.0  # 正規化
            lab_attr = lab_attrs[param] / 10.0
            importance = profile.importance[param]
            
            membership = fuzzy_membership(student_pref, lab_attr, sigma=0.2)
            weighted_sum += membership * importance
            weight_total += importance
        
        environment_score = weighted_sum / weight_total if weight_total > 0 else 0
        
        # 研究分野マッチング
        field_score = calculate_field_match(profile.field_interests, lab_info['field'])
        
        # 総合スコア（環境70% + 分野30%）
        # 実際のシステムでは相互作用があるが、簡易版では線形結合
        total_score = environment_score * 0.7 + field_score * 0.3
        
        results.append(RecommendationResult(
            lab_id=lab_id,
            lab_name=lab_info['name'],
            total_score=total_score,
            field_score=field_score,
            environment_score=environment_score,
            rank=0  # 後で設定
        ))
    
    # スコア降順でソート
    results.sort(key=lambda x: x.total_score, reverse=True)
    
    # 順位を設定
    for i, result in enumerate(results):
        result.rank = i + 1
    
    return results

# =============================================================================
# Phase 2: Primitive Model実装
# =============================================================================

def pm1_field_only(profile: StudentProfile) -> List[RecommendationResult]:
    """
    PM1: 分野一致のみモデル
    
    学生の希望分野と一致するゼミのみを推薦（環境適合度は無視）
    """
    results = []
    
    for lab_id, lab_info in LABS.items():
        # 分野一致度のみで判断
        field_score = calculate_field_match(profile.field_interests, lab_info['field'])
        
        results.append(RecommendationResult(
            lab_id=lab_id,
            lab_name=lab_info['name'],
            total_score=field_score,
            field_score=field_score,
            environment_score=0,  # 考慮しない
            rank=0
        ))
    
    results.sort(key=lambda x: x.total_score, reverse=True)
    for i, result in enumerate(results):
        result.rank = i + 1
    
    return results

def pm2a_popularity_only(profile: StudentProfile) -> List[RecommendationResult]:
    """
    PM2a: 純粋な人気順モデル（従来版）
    
    ゼミの人気度のみで推薦（学生の個性は完全に無視）
    → 全学生に同じ順位で推薦される極端なケース
    """
    results = []
    
    for lab_id, lab_info in LABS.items():
        popularity = lab_info['popularity']
        
        results.append(RecommendationResult(
            lab_id=lab_id,
            lab_name=lab_info['name'],
            total_score=popularity,
            field_score=0,
            environment_score=0,
            rank=0
        ))
    
    results.sort(key=lambda x: x.total_score, reverse=True)
    for i, result in enumerate(results):
        result.rank = i + 1
    
    return results

def pm2b_field_popularity(profile: StudentProfile) -> List[RecommendationResult]:
    """
    PM2b: 分野内人気順モデル（改善版）
    
    学生の興味分野内で人気のゼミを優先的に推薦
    → 「自分の興味分野で人気のゼミを選ぶ」という現実的な行動をシミュレート
    
    計算式: score = popularity × field_weight
      - 興味分野と一致: field_weight = 1.0
      - 興味分野と不一致: field_weight = 0.3
    """
    results = []
    
    for lab_id, lab_info in LABS.items():
        popularity = lab_info['popularity']
        
        # 分野一致なら重み1.0、不一致なら0.3
        if lab_info['field'] in profile.field_interests:
            field_weight = 1.0
        else:
            field_weight = 0.3
        
        # 人気度 × 分野重み
        total_score = popularity * field_weight
        field_score = calculate_field_match(profile.field_interests, lab_info['field'])
        
        results.append(RecommendationResult(
            lab_id=lab_id,
            lab_name=lab_info['name'],
            total_score=total_score,
            field_score=field_score,
            environment_score=0,
            rank=0
        ))
    
    results.sort(key=lambda x: x.total_score, reverse=True)
    for i, result in enumerate(results):
        result.rank = i + 1
    
    return results

def pm2c_popularity_field_blend(profile: StudentProfile) -> List[RecommendationResult]:
    """
    PM2c: 人気度+分野ブレンドモデル
    
    人気度と分野マッチングを重み付きで統合
    → 「人気も見るけど分野も考慮する」という行動をシミュレート
    
    計算式: score = popularity × 0.6 + field_score × 0.4
    """
    results = []
    
    for lab_id, lab_info in LABS.items():
        popularity = lab_info['popularity']
        field_score = calculate_field_match(profile.field_interests, lab_info['field'])
        
        # 人気度60% + 分野40%
        total_score = popularity * 0.6 + field_score * 0.4
        
        results.append(RecommendationResult(
            lab_id=lab_id,
            lab_name=lab_info['name'],
            total_score=total_score,
            field_score=field_score,
            environment_score=0,
            rank=0
        ))
    
    results.sort(key=lambda x: x.total_score, reverse=True)
    for i, result in enumerate(results):
        result.rank = i + 1
    
    return results

def pm3_simple_average(profile: StudentProfile) -> List[RecommendationResult]:
    """
    PM3: 単純平均モデル
    
    11項目の差分の単純平均（重み・ファジィなし）
    """
    results = []
    
    for lab_id, lab_info in LABS.items():
        lab_attrs = LAB_ATTRIBUTES[lab_id]
        
        # 単純な差分の平均
        total_diff = 0
        for param in PARAMETERS:
            diff = abs(profile.preferences[param] - lab_attrs[param])
            total_diff += diff
        
        avg_diff = total_diff / len(PARAMETERS)
        # スコアに変換（差が小さいほど高スコア）
        score = 1 - (avg_diff / 9)  # 最大差は9
        
        results.append(RecommendationResult(
            lab_id=lab_id,
            lab_name=lab_info['name'],
            total_score=score,
            field_score=0,
            environment_score=score,
            rank=0
        ))
    
    results.sort(key=lambda x: x.total_score, reverse=True)
    for i, result in enumerate(results):
        result.rank = i + 1
    
    return results

# =============================================================================
# Phase 3 & 4: セレンディピティ分析
# =============================================================================

def calculate_serendipity_metrics(
    proposed_result: RecommendationResult,
    pm_results: Dict[str, List[RecommendationResult]],
    pm_threshold: int = 10
) -> SerendipityMetrics:
    """
    セレンディピティ指標を計算
    
    Ge et al. (2010)の定義: Serendipity = Unexpected ∩ Useful
    
    Parameters:
        proposed_result: 提案システムの推薦結果
        pm_results: 各PMの推薦結果
        pm_threshold: PMで「推薦されない」と判断する順位閾値
    
    Returns:
        セレンディピティ評価指標
    """
    lab_id = proposed_result.lab_id
    
    # 意外性: PMでは推薦されない（上位に来ない）
    unexpected_count = 0
    for pm_name, pm_ranking in pm_results.items():
        pm_rank = next((r.rank for r in pm_ranking if r.lab_id == lab_id), 31)
        if pm_rank > pm_threshold:
            unexpected_count += 1
    
    unexpectedness = unexpected_count / len(pm_results)
    
    # 関連性: 提案システムでの適合度スコア
    relevance = proposed_result.total_score
    
    # セレンディピティスコア: 意外性 × 関連性
    serendipity_score = unexpectedness * relevance
    
    # セレンディピティ的かどうか（意外でかつ有用）
    is_serendipitous = unexpectedness >= 0.5 and relevance >= 0.7
    
    return SerendipityMetrics(
        serendipity_score=serendipity_score,
        unexpectedness=unexpectedness,
        relevance=relevance,
        is_serendipitous=is_serendipitous
    )

def calculate_coverage(all_recommendations: List[List[RecommendationResult]], k: int = 5) -> Dict:
    """
    カバレッジと公平性指標を計算
    
    Parameters:
        all_recommendations: 全学生の推薦結果
        k: 上位何件を考慮するか
    
    Returns:
        カバレッジ関連の統計
    """
    # 上位K件に登場したゼミをカウント
    top_k_counts = Counter()
    
    for recommendations in all_recommendations:
        for rec in recommendations[:k]:
            top_k_counts[rec.lab_id] += 1
    
    # カバレッジ
    coverage = len(top_k_counts) / len(LABS)
    
    # 出現回数の統計
    counts = list(top_k_counts.values())
    if counts:
        mean_count = np.mean(counts)
        std_count = np.std(counts)
        min_count = min(counts)
        max_count = max(counts)
    else:
        mean_count = std_count = min_count = max_count = 0
    
    # ジニ係数（不平等度）
    if counts and sum(counts) > 0:
        sorted_counts = sorted(counts)
        n = len(sorted_counts)
        cumulative = np.cumsum(sorted_counts)
        gini = (2 * sum((i + 1) * c for i, c in enumerate(sorted_counts)) - (n + 1) * sum(sorted_counts)) / (n * sum(sorted_counts))
    else:
        gini = 0
    
    # 登場しなかったゼミ
    missing_labs = set(LABS.keys()) - set(top_k_counts.keys())
    
    return {
        'coverage': coverage,
        'coverage_count': len(top_k_counts),
        'total_labs': len(LABS),
        'mean_appearance': mean_count,
        'std_appearance': std_count,
        'min_appearance': min_count,
        'max_appearance': max_count,
        'gini_coefficient': gini,
        'missing_labs': list(missing_labs),
        'lab_counts': dict(top_k_counts)
    }

def run_experiment(n_students: int = 1000) -> Dict:
    """
    セレンディピティ評価実験を実行
    
    Parameters:
        n_students: 仮想学生数
    
    Returns:
        実験結果の辞書
    """
    print(f"=" * 70)
    print(f"セレンディピティ評価実験（改善版）")
    print(f"=" * 70)
    
    # Phase 1: プロファイル生成
    print(f"\n[Phase 1] 仮想学生プロファイル生成中... ({n_students}件)")
    profiles = generate_student_profiles(n_students)
    print(f"  → 完了")
    
    # 各モデルで推薦を生成
    print(f"\n[Phase 2-3] 推薦生成・比較分析中...")
    
    all_proposed = []
    all_pm1 = []
    all_pm2a = []  # 純粋な人気順
    all_pm2b = []  # 分野内人気順
    all_pm2c = []  # 人気度+分野ブレンド
    all_pm3 = []
    
    serendipity_cases = []
    all_serendipity_scores = []
    
    for i, profile in enumerate(profiles):
        if (i + 1) % 200 == 0:
            print(f"  処理中: {i + 1}/{n_students}")
        
        # 各モデルで推薦
        proposed = proposed_system_recommend(profile)
        pm1 = pm1_field_only(profile)
        pm2a = pm2a_popularity_only(profile)
        pm2b = pm2b_field_popularity(profile)
        pm2c = pm2c_popularity_field_blend(profile)
        pm3 = pm3_simple_average(profile)
        
        all_proposed.append(proposed)
        all_pm1.append(pm1)
        all_pm2a.append(pm2a)
        all_pm2b.append(pm2b)
        all_pm2c.append(pm2c)
        all_pm3.append(pm3)
        
        # 上位5件についてセレンディピティ分析（PM2bを使用）
        pm_results = {
            'PM1_field': pm1, 
            'PM2b_field_popularity': pm2b,  # 改善版PM2を使用
            'PM3_simple': pm3
        }
        
        for rec in proposed[:5]:
            metrics = calculate_serendipity_metrics(rec, pm_results)
            all_serendipity_scores.append({
                'student_id': profile.id,
                'lab_id': rec.lab_id,
                'lab_name': rec.lab_name,
                'proposed_rank': rec.rank,
                'total_score': rec.total_score,
                'field_score': rec.field_score,
                'serendipity_score': metrics.serendipity_score,
                'unexpectedness': metrics.unexpectedness,
                'relevance': metrics.relevance,
                'is_serendipitous': metrics.is_serendipitous
            })
            
            if metrics.is_serendipitous:
                serendipity_cases.append({
                    'student': profile,
                    'recommendation': rec,
                    'metrics': metrics
                })
    
    print(f"  → 完了")
    
    # Phase 4: 結果分析
    print(f"\n[Phase 4] 結果分析中...")
    
    # カバレッジ分析
    coverage_proposed = calculate_coverage(all_proposed, k=5)
    coverage_pm1 = calculate_coverage(all_pm1, k=5)
    coverage_pm2a = calculate_coverage(all_pm2a, k=5)
    coverage_pm2b = calculate_coverage(all_pm2b, k=5)
    coverage_pm2c = calculate_coverage(all_pm2c, k=5)
    coverage_pm3 = calculate_coverage(all_pm3, k=5)
    
    # セレンディピティ統計
    serendipity_scores = [s['serendipity_score'] for s in all_serendipity_scores]
    serendipitous_count = sum(1 for s in all_serendipity_scores if s['is_serendipitous'])
    
    results = {
        'experiment_info': {
            'n_students': n_students,
            'n_labs': len(LABS),
            'n_parameters': len(PARAMETERS),
            'version': 'improved_pm2'
        },
        'coverage': {
            'proposed': coverage_proposed,
            'pm1_field': coverage_pm1,
            'pm2a_popularity_only': coverage_pm2a,
            'pm2b_field_popularity': coverage_pm2b,
            'pm2c_popularity_field': coverage_pm2c,
            'pm3_simple': coverage_pm3
        },
        'serendipity': {
            'total_evaluations': len(all_serendipity_scores),
            'serendipitous_count': serendipitous_count,
            'serendipity_rate': serendipitous_count / len(all_serendipity_scores) if all_serendipity_scores else 0,
            'mean_score': np.mean(serendipity_scores) if serendipity_scores else 0,
            'std_score': np.std(serendipity_scores) if serendipity_scores else 0,
            'max_score': max(serendipity_scores) if serendipity_scores else 0,
            'min_score': min(serendipity_scores) if serendipity_scores else 0,
            'note': 'Calculated using PM2b (field_popularity) as baseline'
        },
        'detailed_scores': all_serendipity_scores[:100],
        'serendipity_cases_sample': [
            {
                'student_id': case['student'].id,
                'lab_name': case['recommendation'].lab_name,
                'total_score': case['recommendation'].total_score,
                'field_score': case['recommendation'].field_score,
                'unexpectedness': case['metrics'].unexpectedness,
                'serendipity_score': case['metrics'].serendipity_score
            }
            for case in serendipity_cases[:20]
        ]
    }
    
    print(f"  → 完了")
    
    return results

def print_results(results: Dict):
    """結果を整形して表示"""
    print("\n" + "=" * 70)
    print("実験結果サマリー（改善版PM2）")
    print("=" * 70)
    
    info = results['experiment_info']
    print(f"\n■ 実験設定")
    print(f"  仮想学生数: {info['n_students']}")
    print(f"  ゼミ数: {info['n_labs']}")
    print(f"  評価パラメータ数: {info['n_parameters']}")
    
    print(f"\n■ カバレッジ分析（上位5件）")
    print(f"  {'モデル':<25} {'カバレッジ':>10} {'ジニ係数':>10} {'平均出現':>10}")
    print("-" * 60)
    
    for model_name, cov in results['coverage'].items():
        print(f"  {model_name:<25} {cov['coverage']*100:>9.1f}% {cov['gini_coefficient']:>10.3f} {cov['mean_appearance']:>10.1f}")
    
    print(f"\n■ セレンディピティ分析（PM2b基準）")
    ser = results['serendipity']
    print(f"  総評価数: {ser['total_evaluations']}")
    print(f"  セレンディピティ的推薦数: {ser['serendipitous_count']}")
    print(f"  セレンディピティ率: {ser['serendipity_rate']*100:.2f}%")
    print(f"  平均セレンディピティスコア: {ser['mean_score']:.4f}")
    print(f"  標準偏差: {ser['std_score']:.4f}")
    
    if results['serendipity_cases_sample']:
        print(f"\n■ セレンディピティ的推薦の例（上位5件）")
        for i, case in enumerate(results['serendipity_cases_sample'][:5], 1):
            print(f"  {i}. {case['lab_name']}")
            print(f"     総合スコア: {case['total_score']:.3f}, 分野スコア: {case['field_score']:.3f}")
            print(f"     意外性: {case['unexpectedness']:.2f}, セレンディピティ: {case['serendipity_score']:.4f}")

# =============================================================================
# メイン実行
# =============================================================================

if __name__ == "__main__":
    # 実験実行
    results = run_experiment(n_students=1000)
    
    # 結果表示
    print_results(results)
    
    # 結果をJSONファイルに保存
    output_file = "serendipity_experiment_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n結果を {output_file} に保存しました。")