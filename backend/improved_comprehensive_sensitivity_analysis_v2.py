"""
改善版 研究室配属システム - 包括的感度分析システム（交互作用分析対応版）

【主な改善点】
1. field_interests（分野興味度）を感度分析対象に追加
2. research_field_match の順位変化を正確に測定
3. field_interests × research_field_match の交互作用分析
4. 分野選択の重要度を定量化
5. 【新規】全パラメータペアの交互作用分析（Phase 5）

使い方:
  python improved_comprehensive_sensitivity_analysis_v2.py --mode full --no-confirm
  python improved_comprehensive_sensitivity_analysis_v2.py --mode single --lab lab_001
"""

import sys
import os
from typing import Dict, List, Any, Tuple
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from itertools import combinations
import math

# =============================================================================
# 定数定義
# =============================================================================

# 20分野の定義
FIELD_NAMES = [
    'ai_ml', 'image_processing', 'network_security', 'database_systems',
    'embedded_iot', 'education_linguistics', 'natural_science_math',
    'tourism_regional', 'business_decision', 'audio_processing',
    'system_ethics', 'medical_healthcare', 'web_design', 'design_visual',
    'video_animation', 'computer_music', 'game_esports', 'vr_ar_media',
    'philosophy_humanities', 'sports_science'
]

FIELD_LABELS = {
    'ai_ml': '人工知能・機械学習',
    'image_processing': '画像・映像処理',
    'network_security': 'ネットワーク・セキュリティ',
    'database_systems': 'データベース・情報システム',
    'embedded_iot': '組込み・IoT',
    'education_linguistics': '教育・言語学',
    'natural_science_math': '自然科学・数理',
    'tourism_regional': '観光情報・地域システム',
    'business_decision': '経営情報・意思決定支援',
    'audio_processing': '音声・音響情報処理',
    'system_ethics': 'システム運用・情報倫理',
    'medical_healthcare': '医療情報・ヘルスケア',
    'web_design': 'Webデザイン・UI/UX',
    'design_visual': 'デザイン・視覚表現',
    'video_animation': '映像・アニメーション',
    'computer_music': 'コンピュータ音楽・サウンドアート',
    'game_esports': 'ゲーム開発・eスポーツ',
    'vr_ar_media': 'VR/AR・メディアアート',
    'philosophy_humanities': '哲学・人文・環境行動学',
    'sports_science': 'スポーツ・体育科学'
}

# カテゴリマッピング
FIELD_CATEGORIES = {
    "テクノロジー・システム": [
        "ai_ml", "image_processing", "network_security", "database_systems",
        "embedded_iot", "education_linguistics", "natural_science_math",
        "tourism_regional", "business_decision", "audio_processing",
        "system_ethics", "medical_healthcare"
    ],
    "クリエイティブ": ["web_design", "design_visual", "video_animation", "computer_music"],
    "エンターテイメント": ["game_esports", "vr_ar_media"],
    "人文・社会・体育": ["philosophy_humanities", "sports_science"]
}

# 基本12項目
BASIC_CRITERIA = [
    "research_intensity", "advisor_style", "team_work", "workload",
    "theory_practice", "research_field_match", "skill_development",
    "lab_atmosphere", "flexibility", "publication_opportunity",
    "interdisciplinary", "communication_style"
]

# デフォルト重み
DEFAULT_WEIGHTS = {
    "research_intensity": 1.2, "advisor_style": 1.2, "team_work": 1.2,
    "workload": 1.2, "theory_practice": 1.2, "research_field_match": 1.0,
    "skill_development": 1.0, "lab_atmosphere": 1.0, "flexibility": 1.0,
    "publication_opportunity": 1.0, "interdisciplinary": 0.8, "communication_style": 0.8
}


# =============================================================================
# ユーティリティ関数
# =============================================================================

def get_field_category(field_id: str) -> str:
    """分野IDからカテゴリを取得"""
    for category, fields in FIELD_CATEGORIES.items():
        if field_id in fields:
            return category
    return None


def gaussian_similarity(v1: float, v2: float, sigma: float = 0.2) -> float:
    """ガウス類似度計算"""
    v1_norm = (v1 - 1) / 9
    v2_norm = (v2 - 1) / 9
    d = abs(v1_norm - v2_norm)
    return math.exp(-(d ** 2) / (2 * sigma ** 2))


def calculate_basic_score(student_profile: Dict, lab_profile: Dict) -> float:
    """基本項目スコアを計算（research_field_matchを除く11項目）"""
    criteria = [c for c in BASIC_CRITERIA if c != 'research_field_match']
    
    total_score = 0
    total_weight = 0
    
    for criterion in criteria:
        student_val = student_profile.get(criterion, 5.5)
        lab_val = lab_profile.get(criterion, 5.5)
        weight = DEFAULT_WEIGHTS.get(criterion, 1.0)
        priority = student_profile.get(f"{criterion}_priority", 5.0)
        
        sim = gaussian_similarity(student_val, lab_val)
        total_score += sim * priority * weight
        total_weight += priority * weight
    
    return total_score / total_weight if total_weight > 0 else 0


def calculate_field_score(student_field_interests: Dict, lab_field_id: str) -> float:
    """分野スコアを計算"""
    if not lab_field_id:
        return 0.3
    
    lab_category = get_field_category(lab_field_id)
    best_score = 0.3  # 不一致の場合のデフォルト
    
    for field_id, interest in student_field_interests.items():
        interest_norm = interest / 10
        
        if field_id == lab_field_id:
            # 完全一致
            score = interest_norm
        elif get_field_category(field_id) == lab_category:
            # カテゴリ一致
            score = interest_norm * 0.7
        else:
            # 不一致
            score = 0.3
        
        best_score = max(best_score, score)
    
    return best_score


def calculate_final_score(student_profile: Dict, lab_profile: Dict, 
                          student_field_interests: Dict, lab_field_id: str) -> float:
    """最終スコアを計算"""
    research_field_match = student_profile.get('research_field_match', 5)
    alpha = research_field_match / 10
    beta = 1 - alpha
    
    basic_score = calculate_basic_score(student_profile, lab_profile)
    field_score = calculate_field_score(student_field_interests, lab_field_id)
    
    return beta * basic_score + alpha * field_score


# =============================================================================
# 改善版 感度分析クラス
# =============================================================================

class ImprovedSensitivityAnalyzer:
    """
    改善版感度分析クラス
    
    改善点:
    1. field_interestsを分析対象に追加
    2. research_field_matchの順位変化を正確に測定
    3. 交互作用分析
    4. 【新規】全パラメータペアの交互作用分析（Phase 5）
    """
    
    def __init__(self, labs_data: List[Dict]):
        self.labs_data = labs_data
        self.criteria = [c for c in BASIC_CRITERIA if c != 'research_field_match']
        
        # 精度設定
        self.num_samples = 1000
        self.transition_steps = 50
        self.interaction_grid_size = 10  # 交互作用分析のグリッドサイズ
        
    def _get_lab_field_id(self, lab: Dict) -> str:
        """研究室の専門分野IDを取得"""
        if 'field_id' in lab:
            return lab['field_id']
        
        # research_fieldsから推定
        field_mapping = {
            '人工知能': 'ai_ml', '機械学習': 'ai_ml', 'AI': 'ai_ml',
            '画像処理': 'image_processing', '3DCG': 'image_processing',
            'セキュリティ': 'network_security', 'ネットワーク': 'network_security',
            'データベース': 'database_systems', 'データ工学': 'database_systems',
            'IoT': 'embedded_iot', '組み込み': 'embedded_iot',
            '教育': 'education_linguistics', '言語': 'education_linguistics',
            '数学': 'natural_science_math', '統計': 'natural_science_math',
            '観光': 'tourism_regional', '地域': 'tourism_regional',
            '音声': 'audio_processing', '音響': 'audio_processing',
            'システム': 'system_ethics',
            '医療': 'medical_healthcare',
            'Web': 'web_design', 'UI': 'web_design', 'UX': 'web_design',
            'デザイン': 'design_visual', 'イラスト': 'design_visual',
            '映像': 'video_animation', 'アニメ': 'video_animation',
            '音楽': 'computer_music', 'メディアアート': 'computer_music',
            'ゲーム': 'game_esports', 'eスポーツ': 'game_esports',
            'VR': 'vr_ar_media', 'AR': 'vr_ar_media',
            '哲学': 'philosophy_humanities', '芸術学': 'philosophy_humanities',
            'スポーツ': 'sports_science', 'トレーニング': 'sports_science'
        }
        
        for field in lab.get('research_fields', []):
            for keyword, fid in field_mapping.items():
                if keyword in field:
                    return fid
        
        return 'system_ethics'  # デフォルト
    
    def _match_all_labs(self, student_profile: Dict, field_interests: Dict) -> List[Dict]:
        """全研究室とのマッチング"""
        results = []
        
        for lab in self.labs_data:
            lab_field_id = self._get_lab_field_id(lab)
            score = calculate_final_score(
                student_profile, lab, field_interests, lab_field_id
            )
            results.append({
                'lab_id': lab.get('id') or lab.get('lab_id'),
                'lab_name': lab.get('name') or lab.get('lab_name'),
                'field_id': lab_field_id,
                'score': score
            })
        
        results.sort(key=lambda x: -x['score'])
        for i, r in enumerate(results):
            r['rank'] = i + 1
        
        return results
    
    # =========================================================================
    # Phase 1: 基本パラメータ重要度分析（改善版）
    # =========================================================================
    
    def analyze_basic_parameter_importance(self, target_lab_id: str) -> Dict:
        """
        基本12項目のパラメータ重要度分析
        
        改善点: research_field_matchの影響も順位変化で測定
        """
        print(f"\n  [Phase 1] 基本パラメータ重要度分析")
        
        target_lab = next((l for l in self.labs_data if (l.get('id') or l.get('lab_id')) == target_lab_id), None)
        if not target_lab:
            return {}
        
        lab_field_id = self._get_lab_field_id(target_lab)
        
        # ベースラインプロファイル
        base_profile = {c: 5.5 for c in BASIC_CRITERIA}
        for c in BASIC_CRITERIA:
            base_profile[f"{c}_priority"] = 5.0
        
        # 対象研究室の分野に興味を設定
        base_field_interests = {lab_field_id: 8.0}
        
        results = {}
        
        for criterion in BASIC_CRITERIA:
            print(f"    {criterion}...", end=" ", flush=True)
            
            scores = []
            ranks = []
            top_labs = []
            
            test_values = np.linspace(1.0, 10.0, self.transition_steps)
            
            for value in test_values:
                test_profile = base_profile.copy()
                test_profile[criterion] = value
                
                all_results = self._match_all_labs(test_profile, base_field_interests)
                target_result = next((r for r in all_results if r['lab_id'] == target_lab_id), None)
                
                if target_result:
                    scores.append(target_result['score'])
                    ranks.append(target_result['rank'])
                    top_labs.append(all_results[0]['lab_name'])
            
            # 統計計算
            score_range = max(scores) - min(scores) if scores else 0
            rank_range = max(ranks) - min(ranks) if ranks else 0
            optimal_idx = np.argmax(scores) if scores else 0
            
            # 順位遷移点を検出
            transitions = []
            for i in range(1, len(ranks)):
                if ranks[i] != ranks[i-1]:
                    transitions.append({
                        'value': float(test_values[i]),
                        'rank_before': ranks[i-1],
                        'rank_after': ranks[i],
                        'direction': '上昇' if ranks[i] < ranks[i-1] else '下降'
                    })
            
            results[criterion] = {
                'score_range': float(score_range),
                'rank_range': int(rank_range),
                'optimal_value': float(test_values[optimal_idx]),
                'optimal_score': float(scores[optimal_idx]) if scores else 0,
                'num_transitions': len(transitions),
                'transitions': transitions,
                'importance': float(score_range) + float(rank_range) * 0.01  # 複合指標
            }
            
            print(f"変動幅={score_range:.4f}, 順位変動={rank_range}")
        
        # ランキング
        sorted_params = sorted(results.items(), key=lambda x: -x[1]['importance'])
        
        return {
            'parameter_importance': results,
            'top_3_influential': [p[0] for p in sorted_params[:3]],
            'least_influential': [p[0] for p in sorted_params[-3:]]
        }
    
    # =========================================================================
    # Phase 2: research_field_match の詳細分析（新規）
    # =========================================================================
    
    def analyze_research_field_match_impact(self, target_lab_id: str) -> Dict:
        """
        research_field_match（分野重視度）の詳細な影響分析
        
        このパラメータは特殊：
        - 基本項目スコアと分野スコアの重み付け係数として機能
        - 値を変えると順位が大きく変動する可能性がある
        """
        print(f"\n  [Phase 2] research_field_match 詳細分析")
        
        target_lab = next((l for l in self.labs_data if (l.get('id') or l.get('lab_id')) == target_lab_id), None)
        if not target_lab:
            return {}
        
        lab_field_id = self._get_lab_field_id(target_lab)
        
        # ベースラインプロファイル
        base_profile = {c: 5.5 for c in BASIC_CRITERIA}
        for c in BASIC_CRITERIA:
            base_profile[f"{c}_priority"] = 5.0
        
        # 対象研究室の分野に興味を設定
        base_field_interests = {lab_field_id: 8.0}
        
        rfm_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        results_by_rfm = {}
        
        for rfm in rfm_values:
            test_profile = base_profile.copy()
            test_profile['research_field_match'] = rfm
            
            all_results = self._match_all_labs(test_profile, base_field_interests)
            target_result = next((r for r in all_results if r['lab_id'] == target_lab_id), None)
            
            results_by_rfm[rfm] = {
                'rank': target_result['rank'] if target_result else 999,
                'score': target_result['score'] if target_result else 0,
                'top_3': [
                    {'lab_name': r['lab_name'], 'score': r['score'], 'field_id': r['field_id']}
                    for r in all_results[:3]
                ],
                'alpha': rfm / 10,
                'beta': 1 - rfm / 10
            }
            
            print(f"    rfm={rfm}: 順位={results_by_rfm[rfm]['rank']}, スコア={results_by_rfm[rfm]['score']:.4f}")
        
        # 最適なresearch_field_match値を特定
        best_rfm = min(results_by_rfm.items(), key=lambda x: x[1]['rank'])
        worst_rfm = max(results_by_rfm.items(), key=lambda x: x[1]['rank'])
        
        return {
            'results_by_rfm': results_by_rfm,
            'best_rfm': {
                'value': best_rfm[0],
                'rank': best_rfm[1]['rank'],
                'score': best_rfm[1]['score']
            },
            'worst_rfm': {
                'value': worst_rfm[0],
                'rank': worst_rfm[1]['rank'],
                'score': worst_rfm[1]['score']
            },
            'rank_range': worst_rfm[1]['rank'] - best_rfm[1]['rank'],
            'interpretation': self._interpret_rfm_impact(results_by_rfm)
        }
    
    def _interpret_rfm_impact(self, results_by_rfm: Dict) -> str:
        """research_field_matchの影響を解釈"""
        ranks = [r['rank'] for r in results_by_rfm.values()]
        rank_range = max(ranks) - min(ranks)
        
        if rank_range >= 10:
            return "research_field_matchは順位に非常に大きな影響を与える（変動幅10以上）"
        elif rank_range >= 5:
            return "research_field_matchは順位に中程度の影響を与える（変動幅5-9）"
        elif rank_range >= 2:
            return "research_field_matchは順位に小さな影響を与える（変動幅2-4）"
        else:
            return "research_field_matchは順位にほとんど影響を与えない（変動幅1以下）"
    
    # =========================================================================
    # Phase 3: field_interests（分野興味度）の感度分析（新規）
    # =========================================================================
    
    def analyze_field_interests_sensitivity(self, target_lab_id: str) -> Dict:
        """
        field_interests（分野興味度）の感度分析
        
        分析内容:
        1. 各分野への興味度を変化させたときの順位変化
        2. 対象研究室の分野 vs 他分野の影響比較
        3. 分野興味のパターン別分析
        """
        print(f"\n  [Phase 3] field_interests 感度分析")
        
        target_lab = next((l for l in self.labs_data if (l.get('id') or l.get('lab_id')) == target_lab_id), None)
        if not target_lab:
            return {}
        
        lab_field_id = self._get_lab_field_id(target_lab)
        lab_category = get_field_category(lab_field_id)
        
        # ベースラインプロファイル
        base_profile = {c: 5.5 for c in BASIC_CRITERIA}
        for c in BASIC_CRITERIA:
            base_profile[f"{c}_priority"] = 5.0
        base_profile['research_field_match'] = 5  # 50%ずつ
        
        results = {
            'target_field_analysis': {},
            'other_fields_analysis': {},
            'pattern_analysis': {},
            'category_analysis': {}
        }
        
        # 3.1 対象研究室の分野への興味度を変化
        print(f"    対象分野（{lab_field_id}）の興味度を変化...")
        
        target_field_results = []
        for interest in np.linspace(1, 10, 10):
            field_interests = {lab_field_id: interest}
            all_results = self._match_all_labs(base_profile, field_interests)
            target_result = next((r for r in all_results if r['lab_id'] == target_lab_id), None)
            
            target_field_results.append({
                'interest': float(interest),
                'rank': target_result['rank'] if target_result else 999,
                'score': target_result['score'] if target_result else 0,
                'top_lab': all_results[0]['lab_name']
            })
        
        results['target_field_analysis'] = {
            'field_id': lab_field_id,
            'field_label': FIELD_LABELS.get(lab_field_id, lab_field_id),
            'results': target_field_results,
            'rank_range': max(r['rank'] for r in target_field_results) - min(r['rank'] for r in target_field_results),
            'score_range': max(r['score'] for r in target_field_results) - min(r['score'] for r in target_field_results)
        }
        
        # 3.2 他分野への興味度パターン分析
        print(f"    他分野への興味パターン分析...")
        
        patterns = [
            ('対象分野のみ高興味', {lab_field_id: 10}),
            ('対象分野低興味', {lab_field_id: 1}),
            ('同カテゴリ高興味', {f: 8 for f in FIELD_CATEGORIES.get(lab_category, []) if f != lab_field_id}),
            ('異カテゴリ高興味', {f: 8 for cat, fields in FIELD_CATEGORIES.items() if cat != lab_category for f in fields}),
            ('全分野均等', {f: 5 for f in FIELD_NAMES}),
            ('分野興味なし', {}),
        ]
        
        pattern_results = {}
        for pattern_name, field_interests in patterns:
            all_results = self._match_all_labs(base_profile, field_interests)
            target_result = next((r for r in all_results if r['lab_id'] == target_lab_id), None)
            
            pattern_results[pattern_name] = {
                'rank': target_result['rank'] if target_result else 999,
                'score': target_result['score'] if target_result else 0,
                'top_lab': all_results[0]['lab_name'],
                'top_3': [r['lab_name'] for r in all_results[:3]]
            }
            print(f"      {pattern_name}: 順位={pattern_results[pattern_name]['rank']}")
        
        results['pattern_analysis'] = pattern_results
        
        # 3.3 research_field_match × field_interests の交互作用
        print(f"    research_field_match × field_interests 交互作用...")
        
        interaction_results = {}
        for rfm in [1, 5, 10]:
            test_profile = base_profile.copy()
            test_profile['research_field_match'] = rfm
            
            interaction_results[rfm] = {}
            for interest in [1, 5, 10]:
                field_interests = {lab_field_id: interest}
                all_results = self._match_all_labs(test_profile, field_interests)
                target_result = next((r for r in all_results if r['lab_id'] == target_lab_id), None)
                
                interaction_results[rfm][interest] = {
                    'rank': target_result['rank'] if target_result else 999,
                    'score': target_result['score'] if target_result else 0,
                    'top_lab': all_results[0]['lab_name']
                }
        
        results['interaction_analysis'] = interaction_results
        
        return results
    
    # =========================================================================
    # Phase 4: 1位獲得条件分析（改善版）
    # =========================================================================
    
    def analyze_top_rank_conditions(self, target_lab_id: str, num_samples: int = 1000) -> Dict:
        """
        1位獲得条件の分析
        
        改善点:
        - field_interestsもランダムに変化させる
        - research_field_matchの影響を考慮
        """
        print(f"\n  [Phase 4] 1位獲得条件分析（{num_samples}サンプル）")
        
        target_lab = next((l for l in self.labs_data if (l.get('id') or l.get('lab_id')) == target_lab_id), None)
        if not target_lab:
            return {}
        
        lab_field_id = self._get_lab_field_id(target_lab)
        
        top_rank_profiles = []
        all_samples = []
        
        for i in range(num_samples):
            # ランダムプロファイル生成
            random_profile = {c: np.random.uniform(1, 10) for c in BASIC_CRITERIA}
            for c in BASIC_CRITERIA:
                random_profile[f"{c}_priority"] = np.random.uniform(1, 10)
            
            # ランダムな分野興味
            num_interests = np.random.randint(1, 4)
            selected_fields = np.random.choice(FIELD_NAMES, num_interests, replace=False)
            random_field_interests = {f: np.random.uniform(1, 10) for f in selected_fields}
            
            all_results = self._match_all_labs(random_profile, random_field_interests)
            target_result = next((r for r in all_results if r['lab_id'] == target_lab_id), None)
            
            sample_data = {
                'profile': random_profile.copy(),
                'field_interests': random_field_interests.copy(),
                'rank': target_result['rank'] if target_result else 999,
                'score': target_result['score'] if target_result else 0
            }
            all_samples.append(sample_data)
            
            if target_result and target_result['rank'] == 1:
                top_rank_profiles.append(sample_data)
            
            if (i + 1) % 200 == 0:
                print(f"    {i+1}/{num_samples} サンプル完了... (1位獲得: {len(top_rank_profiles)}件)")
        
        # 統計計算
        if top_rank_profiles:
            typical_profile = {}
            for criterion in BASIC_CRITERIA:
                values = [p['profile'][criterion] for p in top_rank_profiles]
                typical_profile[criterion] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
            
            # 分野興味の統計
            field_interest_counts = defaultdict(int)
            field_interest_values = defaultdict(list)
            for p in top_rank_profiles:
                for field, interest in p['field_interests'].items():
                    field_interest_counts[field] += 1
                    field_interest_values[field].append(interest)
            
            typical_field_interests = {
                field: {
                    'count': count,
                    'frequency': count / len(top_rank_profiles),
                    'mean_interest': float(np.mean(field_interest_values[field])) if field in field_interest_values else 0
                }
                for field, count in field_interest_counts.items()
            }
        else:
            typical_profile = {}
            typical_field_interests = {}
        
        return {
            'num_samples': num_samples,
            'num_top_rank': len(top_rank_profiles),
            'top_rank_probability': len(top_rank_profiles) / num_samples,
            'found_top_rank': len(top_rank_profiles) > 0,
            'typical_profile': typical_profile,
            'typical_field_interests': typical_field_interests,
            'target_field_id': lab_field_id,
            'target_field_in_top_rank_ratio': sum(
                1 for p in top_rank_profiles 
                if lab_field_id in p['field_interests']
            ) / len(top_rank_profiles) if top_rank_profiles else 0
        }
    
    # =========================================================================
    # Phase 5: 全パラメータペアの交互作用分析（新規）
    # =========================================================================
    
    def analyze_all_parameter_interactions(self, target_lab_id: str) -> Dict:
        """
        全パラメータペアの交互作用分析
        
        12項目 × 12項目 の全組み合わせ（66通り）で
        2つのパラメータを同時に変化させて交互作用効果を測定
        """
        print(f"\n  [Phase 5] 全パラメータペアの交互作用分析")
        
        target_lab = next((l for l in self.labs_data if (l.get('id') or l.get('lab_id')) == target_lab_id), None)
        if not target_lab:
            return {}
        
        lab_field_id = self._get_lab_field_id(target_lab)
        
        # ベースラインプロファイル
        base_profile = {c: 5.5 for c in BASIC_CRITERIA}
        for c in BASIC_CRITERIA:
            base_profile[f"{c}_priority"] = 5.0
        
        # 対象研究室の分野に興味を設定
        base_field_interests = {lab_field_id: 8.0}
        
        # 全パラメータペアの組み合わせを生成
        all_pairs = list(combinations(BASIC_CRITERIA, 2))
        total_pairs = len(all_pairs)
        
        print(f"    分析対象ペア数: {total_pairs}組")
        
        interaction_results = {}
        grid_size = self.interaction_grid_size
        test_values = np.linspace(1.0, 10.0, grid_size)
        
        for pair_idx, (param_a, param_b) in enumerate(all_pairs, 1):
            pair_key = f"{param_a}_x_{param_b}"
            print(f"    [{pair_idx}/{total_pairs}] {param_a} × {param_b}...", end=" ", flush=True)
            
            # グリッド探索
            score_grid = np.zeros((grid_size, grid_size))
            rank_grid = np.zeros((grid_size, grid_size), dtype=int)
            
            for i, val_a in enumerate(test_values):
                for j, val_b in enumerate(test_values):
                    test_profile = base_profile.copy()
                    test_profile[param_a] = val_a
                    test_profile[param_b] = val_b
                    
                    all_results = self._match_all_labs(test_profile, base_field_interests)
                    target_result = next((r for r in all_results if r['lab_id'] == target_lab_id), None)
                    
                    if target_result:
                        score_grid[i, j] = target_result['score']
                        rank_grid[i, j] = target_result['rank']
            
            # 主効果の計算（各パラメータ単独の影響）
            # param_aの主効果: param_bを中央値に固定したときの変動
            mid_idx = grid_size // 2
            main_effect_a_scores = score_grid[:, mid_idx]
            main_effect_b_scores = score_grid[mid_idx, :]
            
            main_effect_a = float(np.max(main_effect_a_scores) - np.min(main_effect_a_scores))
            main_effect_b = float(np.max(main_effect_b_scores) - np.min(main_effect_b_scores))
            
            # 交互作用効果の計算
            # 交互作用 = 実際のスコア変動 - (主効果Aによる期待値 + 主効果Bによる期待値)
            baseline_score = score_grid[mid_idx, mid_idx]
            
            # 各コーナーでの交互作用効果を計算
            corners = [
                (0, 0, "low_low"),
                (0, grid_size-1, "low_high"),
                (grid_size-1, 0, "high_low"),
                (grid_size-1, grid_size-1, "high_high")
            ]
            
            corner_interactions = {}
            for i, j, corner_name in corners:
                actual_score = score_grid[i, j]
                expected_from_a = main_effect_a_scores[i] - baseline_score
                expected_from_b = main_effect_b_scores[j] - baseline_score
                expected_score = baseline_score + expected_from_a + expected_from_b
                interaction_effect = actual_score - expected_score
                corner_interactions[corner_name] = {
                    'actual_score': float(actual_score),
                    'expected_score': float(expected_score),
                    'interaction_effect': float(interaction_effect),
                    'param_a_value': float(test_values[i]),
                    'param_b_value': float(test_values[j])
                }
            
            # 交互作用の強さを定量化
            interaction_effects = [abs(c['interaction_effect']) for c in corner_interactions.values()]
            avg_interaction_strength = float(np.mean(interaction_effects))
            max_interaction_strength = float(np.max(interaction_effects))
            
            # スコアと順位の統計
            score_min = float(np.min(score_grid))
            score_max = float(np.max(score_grid))
            score_range = score_max - score_min
            
            rank_min = int(np.min(rank_grid))
            rank_max = int(np.max(rank_grid))
            rank_range = rank_max - rank_min
            
            # 最適な組み合わせを特定
            best_idx = np.unravel_index(np.argmax(score_grid), score_grid.shape)
            worst_idx = np.unravel_index(np.argmin(score_grid), score_grid.shape)
            
            # 交互作用のタイプを判定
            interaction_type = self._classify_interaction_type(corner_interactions)
            
            interaction_results[pair_key] = {
                'param_a': param_a,
                'param_b': param_b,
                'main_effect_a': main_effect_a,
                'main_effect_b': main_effect_b,
                'avg_interaction_strength': avg_interaction_strength,
                'max_interaction_strength': max_interaction_strength,
                'interaction_type': interaction_type,
                'corner_interactions': corner_interactions,
                'score_statistics': {
                    'min': score_min,
                    'max': score_max,
                    'range': score_range,
                    'mean': float(np.mean(score_grid)),
                    'std': float(np.std(score_grid))
                },
                'rank_statistics': {
                    'min': rank_min,
                    'max': rank_max,
                    'range': rank_range,
                    'mean': float(np.mean(rank_grid)),
                    'std': float(np.std(rank_grid))
                },
                'optimal_combination': {
                    'param_a_value': float(test_values[best_idx[0]]),
                    'param_b_value': float(test_values[best_idx[1]]),
                    'score': float(score_grid[best_idx]),
                    'rank': int(rank_grid[best_idx])
                },
                'worst_combination': {
                    'param_a_value': float(test_values[worst_idx[0]]),
                    'param_b_value': float(test_values[worst_idx[1]]),
                    'score': float(score_grid[worst_idx]),
                    'rank': int(rank_grid[worst_idx])
                },
                'score_grid': score_grid.tolist(),
                'rank_grid': rank_grid.tolist(),
                'test_values': test_values.tolist()
            }
            
            print(f"交互作用強度={avg_interaction_strength:.4f}, タイプ={interaction_type}")
        
        # グローバルサマリー
        summary = self._generate_interaction_summary(interaction_results)
        
        return {
            'total_pairs_analyzed': total_pairs,
            'grid_size': grid_size,
            'pair_results': interaction_results,
            'summary': summary
        }
    
    def _classify_interaction_type(self, corner_interactions: Dict) -> str:
        """交互作用のタイプを分類"""
        effects = corner_interactions
        
        # 相乗効果（synergistic）: 両方高い時に期待以上のスコア
        high_high_effect = effects['high_high']['interaction_effect']
        
        # 拮抗効果（antagonistic）: 両方高い時に期待以下のスコア
        
        # 閾値
        threshold = 0.01
        
        if high_high_effect > threshold:
            return "synergistic"  # 相乗効果
        elif high_high_effect < -threshold:
            return "antagonistic"  # 拮抗効果
        else:
            # さらに詳細な分類
            low_low = effects['low_low']['interaction_effect']
            low_high = effects['low_high']['interaction_effect']
            high_low = effects['high_low']['interaction_effect']
            
            if abs(low_high) > threshold or abs(high_low) > threshold:
                return "crossover"  # 交差型交互作用
            else:
                return "additive"  # 加法的（交互作用なし）
    
    def _generate_interaction_summary(self, interaction_results: Dict) -> Dict:
        """交互作用分析のサマリーを生成"""
        # 交互作用強度でソート
        sorted_by_strength = sorted(
            interaction_results.items(),
            key=lambda x: x[1]['avg_interaction_strength'],
            reverse=True
        )
        
        # 順位変動幅でソート
        sorted_by_rank_range = sorted(
            interaction_results.items(),
            key=lambda x: x[1]['rank_statistics']['range'],
            reverse=True
        )
        
        # 交互作用タイプ別の集計
        type_counts = defaultdict(int)
        for pair_key, data in interaction_results.items():
            type_counts[data['interaction_type']] += 1
        
        # 強い交互作用を持つペア（上位10%）
        top_10_percent = max(1, len(sorted_by_strength) // 10)
        strong_interactions = [
            {
                'pair': item[0],
                'param_a': item[1]['param_a'],
                'param_b': item[1]['param_b'],
                'strength': item[1]['avg_interaction_strength'],
                'type': item[1]['interaction_type']
            }
            for item in sorted_by_strength[:top_10_percent]
        ]
        
        # 順位に大きく影響するペア（上位10%）
        rank_influential = [
            {
                'pair': item[0],
                'param_a': item[1]['param_a'],
                'param_b': item[1]['param_b'],
                'rank_range': item[1]['rank_statistics']['range'],
                'best_rank': item[1]['rank_statistics']['min'],
                'worst_rank': item[1]['rank_statistics']['max']
            }
            for item in sorted_by_rank_range[:top_10_percent]
        ]
        
        return {
            'total_pairs': len(interaction_results),
            'interaction_type_distribution': dict(type_counts),
            'top_strong_interactions': strong_interactions,
            'top_rank_influential_pairs': rank_influential,
            'avg_interaction_strength_overall': float(np.mean([
                r['avg_interaction_strength'] for r in interaction_results.values()
            ])),
            'avg_rank_range_overall': float(np.mean([
                r['rank_statistics']['range'] for r in interaction_results.values()
            ]))
        }
    
    # =========================================================================
    # 統合分析
    # =========================================================================
    
    def comprehensive_analysis(self, target_lab_id: str) -> Dict:
        """包括的分析の実行"""
        target_lab = next((l for l in self.labs_data if (l.get('id') or l.get('lab_id')) == target_lab_id), None)
        if not target_lab:
            print(f"研究室 {target_lab_id} が見つかりません")
            return {}
        
        lab_name = target_lab.get('name') or target_lab.get('lab_name')
        
        print(f"\n{'='*70}")
        print(f"研究室: {lab_name} ({target_lab_id})")
        print(f"{'='*70}")
        
        results = {
            'lab_id': target_lab_id,
            'lab_name': lab_name,
            'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'phase1_basic_parameter_importance': self.analyze_basic_parameter_importance(target_lab_id),
            'phase2_research_field_match_impact': self.analyze_research_field_match_impact(target_lab_id),
            'phase3_field_interests_sensitivity': self.analyze_field_interests_sensitivity(target_lab_id),
            'phase4_top_rank_conditions': self.analyze_top_rank_conditions(target_lab_id, self.num_samples),
            'phase5_parameter_interactions': self.analyze_all_parameter_interactions(target_lab_id)
        }
        
        # サマリー生成
        results['summary'] = self._generate_summary(results)
        
        return results
    
    def _generate_summary(self, results: Dict) -> Dict:
        """分析サマリーを生成"""
        phase1 = results.get('phase1_basic_parameter_importance', {})
        phase2 = results.get('phase2_research_field_match_impact', {})
        phase3 = results.get('phase3_field_interests_sensitivity', {})
        phase4 = results.get('phase4_top_rank_conditions', {})
        phase5 = results.get('phase5_parameter_interactions', {})
        
        return {
            'top_3_influential_basic_params': phase1.get('top_3_influential', []),
            'research_field_match_impact': {
                'rank_range': phase2.get('rank_range', 0),
                'best_value': phase2.get('best_rfm', {}).get('value'),
                'interpretation': phase2.get('interpretation', '')
            },
            'field_interests_impact': {
                'target_field_rank_range': phase3.get('target_field_analysis', {}).get('rank_range', 0),
                'target_field': phase3.get('target_field_analysis', {}).get('field_id')
            },
            'top_rank_probability': phase4.get('top_rank_probability', 0),
            'can_achieve_top_rank': phase4.get('found_top_rank', False),
            'interaction_analysis': {
                'total_pairs_analyzed': phase5.get('total_pairs_analyzed', 0),
                'interaction_type_distribution': phase5.get('summary', {}).get('interaction_type_distribution', {}),
                'top_strong_interactions': phase5.get('summary', {}).get('top_strong_interactions', []),
                'top_rank_influential_pairs': phase5.get('summary', {}).get('top_rank_influential_pairs', []),
                'avg_interaction_strength': phase5.get('summary', {}).get('avg_interaction_strength_overall', 0)
            },
            'recommendations': self._generate_recommendations(results)
        }
    
    def _generate_recommendations(self, results: Dict) -> List[str]:
        """推奨事項を生成"""
        recommendations = []
        
        phase2 = results.get('phase2_research_field_match_impact', {})
        phase4 = results.get('phase4_top_rank_conditions', {})
        phase5 = results.get('phase5_parameter_interactions', {})
        
        # 1位獲得可能性
        if phase4.get('found_top_rank'):
            prob = phase4.get('top_rank_probability', 0) * 100
            recommendations.append(f"この研究室は1位獲得が可能です（確率: {prob:.1f}%）")
            
            # 最適なresearch_field_match
            best_rfm = phase2.get('best_rfm', {}).get('value')
            if best_rfm:
                recommendations.append(f"research_field_match の推奨値: {best_rfm}")
            
            # 分野興味の推奨
            target_field = phase4.get('target_field_id')
            if target_field:
                target_ratio = phase4.get('target_field_in_top_rank_ratio', 0) * 100
                recommendations.append(f"対象分野（{target_field}）への興味が重要（1位獲得時の{target_ratio:.0f}%で選択）")
        else:
            recommendations.append("この研究室は1位獲得が困難です")
        
        # 交互作用に基づく推奨
        strong_interactions = phase5.get('summary', {}).get('top_strong_interactions', [])
        if strong_interactions:
            top_interaction = strong_interactions[0]
            recommendations.append(
                f"最も強い交互作用: {top_interaction['param_a']} × {top_interaction['param_b']} "
                f"（強度: {top_interaction['strength']:.4f}, タイプ: {top_interaction['type']}）"
            )
        
        rank_influential = phase5.get('summary', {}).get('top_rank_influential_pairs', [])
        if rank_influential:
            top_pair = rank_influential[0]
            recommendations.append(
                f"順位に最も影響するペア: {top_pair['param_a']} × {top_pair['param_b']} "
                f"（順位変動幅: {top_pair['rank_range']}）"
            )
        
        return recommendations
    
    # =========================================================================
    # 全研究室分析
    # =========================================================================
    
    def analyze_all_labs(self, output_dir: str = "improved_sensitivity_results_v2") -> Dict:
        """全研究室の分析"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        all_results = {}
        
        for i, lab in enumerate(self.labs_data, 1):
            lab_id = lab.get('id') or lab.get('lab_id')
            lab_name = lab.get('name') or lab.get('lab_name')
            
            print(f"\n[{i}/{len(self.labs_data)}] {lab_name} ({lab_id})")
            
            results = self.comprehensive_analysis(lab_id)
            all_results[lab_id] = results
            
            # 個別保存
            lab_dir = output_path / lab_id
            lab_dir.mkdir(parents=True, exist_ok=True)
            
            with open(lab_dir / f"{lab_id}_analysis.json", 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        
        # マスターレポート
        master_report = {
            'total_labs': len(self.labs_data),
            'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'individual_reports': all_results,
            'global_summary': self._generate_global_summary(all_results)
        }
        
        with open(output_path / "master_report.json", 'w', encoding='utf-8') as f:
            json.dump(master_report, f, ensure_ascii=False, indent=2)
        
        print(f"\n{'='*70}")
        print(f"分析完了！ 結果は {output_path} に保存されました")
        print(f"{'='*70}")
        
        return master_report
    
    def _generate_global_summary(self, all_results: Dict) -> Dict:
        """全体サマリーを生成"""
        # 各研究室の最重要パラメータを集計
        param_frequency = defaultdict(int)
        
        for results in all_results.values():
            top_3 = results.get('summary', {}).get('top_3_influential_basic_params', [])
            for param in top_3:
                param_frequency[param] += 1
        
        # research_field_match の影響度集計
        rfm_impact_sum = 0
        rfm_count = 0
        for results in all_results.values():
            rfm_range = results.get('phase2_research_field_match_impact', {}).get('rank_range', 0)
            if rfm_range:
                rfm_impact_sum += rfm_range
                rfm_count += 1
        
        # 交互作用の全体統計
        interaction_type_totals = defaultdict(int)
        all_strong_interactions = []
        
        for lab_id, results in all_results.items():
            phase5 = results.get('phase5_parameter_interactions', {})
            summary = phase5.get('summary', {})
            
            for itype, count in summary.get('interaction_type_distribution', {}).items():
                interaction_type_totals[itype] += count
            
            for interaction in summary.get('top_strong_interactions', []):
                all_strong_interactions.append({
                    'lab_id': lab_id,
                    **interaction
                })
        
        # 最も頻繁に強い交互作用を示すパラメータペア
        pair_strength_sum = defaultdict(float)
        pair_count = defaultdict(int)
        
        for results in all_results.values():
            phase5 = results.get('phase5_parameter_interactions', {})
            pair_results = phase5.get('pair_results', {})
            
            for pair_key, data in pair_results.items():
                pair_strength_sum[pair_key] += data.get('avg_interaction_strength', 0)
                pair_count[pair_key] += 1
        
        avg_pair_strength = {
            pair: pair_strength_sum[pair] / pair_count[pair]
            for pair in pair_strength_sum
        }
        
        top_pairs_globally = sorted(
            avg_pair_strength.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return {
            'globally_influential_parameters': sorted(
                param_frequency.items(),
                key=lambda x: x[1],
                reverse=True
            ),
            'average_rfm_rank_range': rfm_impact_sum / rfm_count if rfm_count > 0 else 0,
            'labs_with_top_rank_potential': sum(
                1 for r in all_results.values()
                if r.get('summary', {}).get('can_achieve_top_rank', False)
            ),
            'total_labs_analyzed': len(all_results),
            'interaction_analysis_global': {
                'total_interaction_type_distribution': dict(interaction_type_totals),
                'top_10_strongest_interaction_pairs_globally': [
                    {'pair': pair, 'avg_strength': strength}
                    for pair, strength in top_pairs_globally
                ]
            }
        }


# =============================================================================
# メイン実行
# =============================================================================

def load_labs_data() -> List[Dict]:
    """研究室データを読み込み"""
    possible_paths = [
        "data/labs_database.json",
        "backend/data/labs_database.json",
        "../data/labs_database.json",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"研究室データを読み込み: {path}")
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, dict):
                labs = data.get("labs", [])
            else:
                labs = data
            
            # フラット化
            flattened = []
            for lab in labs:
                flat_lab = lab.copy()
                if "features" in lab:
                    for key, value in lab["features"].items():
                        flat_lab[key] = value
                flattened.append(flat_lab)
            
            return flattened
    
    raise FileNotFoundError("研究室データベースが見つかりません")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="改善版感度分析システム（交互作用分析対応）")
    parser.add_argument('--mode', choices=['full', 'single'], required=True)
    parser.add_argument('--lab', type=str, help='単一研究室モードのlab_id')
    parser.add_argument('--no-confirm', action='store_true')
    parser.add_argument('--samples', type=int, default=1000, help='サンプル数')
    parser.add_argument('--grid-size', type=int, default=10, help='交互作用分析のグリッドサイズ')
    
    args = parser.parse_args()
    
    # データ読み込み
    labs_data = load_labs_data()
    print(f"✅ {len(labs_data)}研究室を読み込みました")
    
    # アナライザー初期化
    analyzer = ImprovedSensitivityAnalyzer(labs_data)
    analyzer.num_samples = args.samples
    analyzer.interaction_grid_size = args.grid_size
    
    if args.mode == 'single':
        if not args.lab:
            print("❌ --lab オプションで研究室IDを指定してください")
            return 1
        
        results = analyzer.comprehensive_analysis(args.lab)
        
        # 保存
        output_file = f"improved_sensitivity_v2_{args.lab}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n結果を {output_file} に保存しました")
        
    elif args.mode == 'full':
        if not args.no_confirm:
            print(f"\n全{len(labs_data)}研究室の分析を開始します。")
            print(f"交互作用分析: 66ペア × {args.grid_size}×{args.grid_size}グリッド")
            input("準備ができたらEnterキーを押してください...")
        
        analyzer.analyze_all_labs()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())