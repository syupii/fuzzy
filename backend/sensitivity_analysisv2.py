#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完全版 感度分析 - 全データ記録版 v2
11項目すべてを1-10で変動し、すべての値・順位・スコアを記録

出力:
  - sensitivity_all_data_report.txt  (詳細レポート)
  - sensitivity_all_data.json        (JSON形式・全データ)
  - sensitivity_all_data.csv         (CSV形式・Excel対応)

使い方:
  python sensitivity_analysis_all_data_v2.py --mode full --no-confirm
  python sensitivity_analysis_all_data_v2.py --mode debug
"""

import json
import math
import sys
import os
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# =============================================================================
# 定数定義
# =============================================================================

# 基本11項目
BASIC_CRITERIA = [
    "research_intensity",      # 研究強度
    "advisor_style",           # 指導スタイル
    "team_work",               # チームワーク
    "workload",                # ワークロード
    "theory_practice",         # 理論/実践
    "skill_development",       # スキル開発
    "lab_atmosphere",          # 研究室雰囲気
    "flexibility",             # 柔軟性
    "publication_opportunity", # 発表機会
    "interdisciplinary",       # 学際性
    "communication_style"      # コミュニケーション
]

# 日本語名
PARAM_NAMES_JA = {
    'research_intensity': '研究強度',
    'advisor_style': '指導スタイル',
    'team_work': 'チームワーク',
    'workload': 'ワークロード',
    'theory_practice': '理論/実践',
    'skill_development': 'スキル開発',
    'lab_atmosphere': '研究室雰囲気',
    'flexibility': '柔軟性',
    'publication_opportunity': '発表機会',
    'interdisciplinary': '学際性',
    'communication_style': 'コミュニケーション'
}

# 分野カテゴリ
FIELD_CATEGORIES = {
    "テクノロジー・システム": [
        "ai_ml", "image_processing", "network_security", "database_systems",
        "embedded_iot", "education_linguistics", "natural_science_math",
        "tourism_regional", "business_decision", "audio_processing",
        "system_ethics", "medical_healthcare", "software_dev", "data_science_math"
    ],
    "クリエイティブ": [
        "web_design", "design_visual", "video_animation", "computer_music",
        "graphic_visual", "web_design_uiux", "illustration_art", "video_film"
    ],
    "エンターテイメント": [
        "game_esports", "vr_ar_media", "game_dev", "media_art"
    ],
    "人文・社会・体育": [
        "philosophy_humanities", "sports_science", "english_humanities",
        "japanese_education", "korean_studies", "natural_science"
    ]
}

# ガウス類似度のσ値
SIGMA = 0.2

# テスト値の範囲
TEST_VALUES = list(range(1, 11))  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


# =============================================================================
# ユーティリティ関数
# =============================================================================

def get_field_category(field_id: str) -> str:
    """分野IDからカテゴリを取得"""
    for category, fields in FIELD_CATEGORIES.items():
        if field_id in fields:
            return category
    # 不明な場合はキーワードから推測
    field_lower = field_id.lower()
    if any(kw in field_lower for kw in ['design', 'visual', 'art', 'graphic', 'illustration', 'film']):
        return "クリエイティブ"
    if any(kw in field_lower for kw in ['game', 'vr', 'ar', 'media']):
        return "エンターテイメント"
    if any(kw in field_lower for kw in ['human', 'education', 'language', 'sport', 'science', 'korean', 'japanese', 'english']):
        return "人文・社会・体育"
    return "テクノロジー・システム"


def gaussian_similarity(v1_norm: float, v2_norm: float, sigma: float = SIGMA) -> float:
    """ガウス類似度計算（0-1正規化された値を受け取る）"""
    diff = abs(v1_norm - v2_norm)
    return math.exp(-(diff ** 2) / (2 * sigma ** 2))


def get_lab_field_id(lab: Dict) -> str:
    """研究室のfield_idを取得"""
    if 'field_id' in lab:
        return lab['field_id']
    
    # research_fieldsからキーワードで推測
    mapping = {
        '人工知能': 'ai_ml', '機械学習': 'ai_ml', 'AI': 'ai_ml',
        '画像': 'image_processing', '3DCG': 'image_processing', '映像処理': 'image_processing',
        'セキュリティ': 'network_security', 'ネットワーク': 'network_security',
        'データベース': 'database_systems', 'データ工学': 'database_systems',
        'IoT': 'embedded_iot', '組み込み': 'embedded_iot',
        '教育': 'education_linguistics', '言語': 'education_linguistics',
        '数学': 'natural_science_math', '統計': 'natural_science_math',
        '観光': 'tourism_regional', '地域': 'tourism_regional',
        '音声': 'audio_processing', '音響': 'audio_processing',
        'システム': 'system_ethics', '医療': 'medical_healthcare',
        'Web': 'web_design', 'UI': 'web_design', 'UX': 'web_design',
        'デザイン': 'design_visual', 'イラスト': 'design_visual',
        '映像': 'video_animation', 'アニメ': 'video_animation',
        '音楽': 'computer_music', 'サウンド': 'computer_music',
        'ゲーム': 'game_esports', 'eスポーツ': 'game_esports',
        'VR': 'vr_ar_media', 'AR': 'vr_ar_media', 'メディアアート': 'vr_ar_media',
        '哲学': 'philosophy_humanities', '芸術学': 'philosophy_humanities',
        'スポーツ': 'sports_science', 'トレーニング': 'sports_science'
    }
    
    for field in lab.get('research_fields', []):
        for keyword, fid in mapping.items():
            if keyword in field:
                return fid
    
    return 'system_ethics'


def get_lab_value(lab: Dict, criterion: str) -> float:
    """研究室のパラメータ値を取得（1-10スケール）"""
    # 直接存在する場合
    if criterion in lab:
        return float(lab[criterion])
    # features内に存在する場合
    if 'features' in lab and criterion in lab['features']:
        return float(lab['features'][criterion])
    # デフォルト値
    return 5.5


# =============================================================================
# マッチングクラス（遺伝的アルゴリズムと同じロジック）
# =============================================================================

class LabMatcher:
    """研究室マッチング評価器"""
    
    def __init__(self, labs: List[Dict]):
        self.labs = labs
        self.lab_names = []
        self.lab_data = {}
        
        # 各研究室のデータを事前計算
        for lab in labs:
            name = lab.get('name') or lab.get('lab_name')
            self.lab_names.append(name)
            self.lab_data[name] = {
                'id': lab.get('id') or lab.get('lab_id'),
                'field_id': get_lab_field_id(lab),
                'criteria': {c: get_lab_value(lab, c) for c in BASIC_CRITERIA}
            }
    
    def calculate_basic_score(self, student: Dict, lab_name: str) -> float:
        """基本スコア計算（11項目）"""
        lab = self.lab_data.get(lab_name)
        if not lab:
            return 0
        
        total_score = 0
        total_weight = 0
        
        for criterion in BASIC_CRITERIA:
            # 学生の値（1-10 → 0-1正規化）
            student_val = (float(student.get(criterion, 5.5)) - 1) / 9
            
            # 学生の重要度（GAと同じ: (priority-1)/9 + 0.1）
            priority = float(student.get(f"{criterion}_priority", 5.0))
            weight = (priority - 1) / 9 + 0.1
            
            # 研究室の値（1-10 → 0-1正規化）
            lab_val = (lab['criteria'][criterion] - 1) / 9
            
            # ガウス類似度
            sim = gaussian_similarity(student_val, lab_val)
            
            total_score += sim * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0
    
    def calculate_field_score(self, field_interests: Dict, lab_name: str) -> float:
        """分野スコア計算"""
        lab = self.lab_data.get(lab_name)
        if not lab or not field_interests:
            return 0.3
        
        lab_field_id = lab['field_id']
        lab_category = get_field_category(lab_field_id)
        best_score = 0.3
        
        for field_id, interest in field_interests.items():
            # 興味度（1-10 → 0-1正規化）
            interest_norm = float(interest) / 10 if float(interest) > 1 else float(interest)
            
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
    
    def calculate_final_score(self, student: Dict, field_interests: Dict, lab_name: str) -> Tuple[float, float, float]:
        """最終スコア計算（基本スコア、分野スコア、最終スコアを返す）"""
        basic_score = self.calculate_basic_score(student, lab_name)
        field_score = self.calculate_field_score(field_interests, lab_name)
        
        # research_field_matchによる重み付け
        rfm = float(student.get('research_field_match', 5))
        alpha = (rfm - 1) / 9  # 0-1正規化
        beta = 1 - alpha
        
        final_score = beta * basic_score + alpha * field_score
        
        return basic_score, field_score, final_score
    
    def get_full_ranking(self, student: Dict, field_interests: Dict) -> List[Dict]:
        """全研究室のランキングを取得（詳細情報付き）"""
        results = []
        
        for lab_name in self.lab_names:
            basic, field, final = self.calculate_final_score(student, field_interests, lab_name)
            results.append({
                'lab_name': lab_name,
                'lab_id': self.lab_data[lab_name]['id'],
                'field_id': self.lab_data[lab_name]['field_id'],
                'basic_score': basic,
                'field_score': field,
                'final_score': final
            })
        
        # スコアでソート
        results.sort(key=lambda x: -x['final_score'])
        
        # 順位を付与
        for i, r in enumerate(results):
            r['rank'] = i + 1
        
        return results
    
    def find_lab_result(self, ranking: List[Dict], target_name: str) -> Dict:
        """ランキングから対象ゼミの結果を取得"""
        for r in ranking:
            if target_name in r['lab_name'] or r['lab_name'] in target_name:
                return r
        return None


# =============================================================================
# 感度分析クラス
# =============================================================================

class CompleteSensitivityAnalyzer:
    """完全版感度分析（全データ記録）"""
    
    def __init__(self, labs: List[Dict], optimal_solutions: List[Dict]):
        self.labs = labs
        self.optimal_solutions = optimal_solutions
        self.matcher = LabMatcher(labs)
    
    def get_field_interests(self, optimal_data: Dict) -> Dict:
        """最適解からfield_interestsを取得"""
        profile = optimal_data.get('optimal_student_profile', {})
        
        # プロファイル内にある場合
        if 'field_interests' in profile:
            return profile['field_interests']
        
        # 最適解データ直下にある場合
        if 'field_interests' in optimal_data:
            return optimal_data['field_interests']
        
        # デフォルト：対象ゼミの分野に興味8.0
        lab_name = optimal_data.get('lab_name', '')
        lab_data = self.matcher.lab_data.get(lab_name, {})
        lab_field_id = lab_data.get('field_id', 'system_ethics')
        return {lab_field_id: 8.0}
    
    def analyze_single_lab(self, optimal_data: Dict) -> Dict:
        """
        単一ゼミの完全感度分析
        - 11項目すべて
        - 1-10のすべての値
        - 順位・スコア・上位ゼミをすべて記録
        """
        
        lab_name = optimal_data.get('lab_name', '')
        optimal_profile = optimal_data.get('optimal_student_profile', {})
        
        if not optimal_profile:
            return {
                'lab_name': lab_name,
                'error': 'No optimal profile found'
            }
        
        field_interests = self.get_field_interests(optimal_data)
        
        # 基準順位（最適解での順位）
        base_ranking = self.matcher.get_full_ranking(optimal_profile, field_interests)
        base_result = self.matcher.find_lab_result(base_ranking, lab_name)
        
        if not base_result:
            return {
                'lab_name': lab_name,
                'error': f'Lab not found in ranking'
            }
        
        # 結果格納用
        results = {
            'lab_name': lab_name,
            'lab_id': base_result['lab_id'],
            'lab_field_id': base_result['field_id'],
            'field_interests': field_interests,
            'base_rank': base_result['rank'],
            'base_score': round(base_result['final_score'], 6),
            'base_basic_score': round(base_result['basic_score'], 6),
            'base_field_score': round(base_result['field_score'], 6),
            'base_top5': [
                {
                    'rank': r['rank'],
                    'lab_name': r['lab_name'],
                    'score': round(r['final_score'], 6)
                }
                for r in base_ranking[:5]
            ],
            'parameters': {},
            'summary': {
                'total_params': len(BASIC_CRITERIA),
                'params_with_variation': 0,
                'max_rank_drop': 0,
                'most_sensitive_param': None
            }
        }
        
        # 11項目すべてを分析
        for param in BASIC_CRITERIA:
            original_value = float(optimal_profile.get(param, 5.5))
            original_priority = float(optimal_profile.get(f"{param}_priority", 5.0))
            
            param_result = {
                'param_name_ja': PARAM_NAMES_JA.get(param, param),
                'original_value': round(original_value, 4),
                'original_priority': round(original_priority, 4),
                'lab_value': self.matcher.lab_data[lab_name]['criteria'][param],
                'test_results': {},  # 値 -> 詳細結果
                'ranks': [],         # [1-10]の順位リスト
                'scores': [],        # [1-10]のスコアリスト
                'basic_scores': [],  # [1-10]の基本スコアリスト
                'min_rank': 999,
                'max_rank': 0,
                'best_value': None,
                'worst_value': None,
                'rank_range': 0,
                'has_variation': False
            }
            
            # 1から10まですべての値でテスト
            for test_value in TEST_VALUES:
                # プロファイルをコピーして値を変更
                modified_profile = optimal_profile.copy()
                modified_profile[param] = float(test_value)
                
                # ランキング計算
                new_ranking = self.matcher.get_full_ranking(modified_profile, field_interests)
                new_result = self.matcher.find_lab_result(new_ranking, lab_name)
                
                if not new_result:
                    continue
                
                new_rank = new_result['rank']
                new_score = new_result['final_score']
                new_basic = new_result['basic_score']
                rank_change = new_rank - base_result['rank']
                
                # 上位に来たゼミを特定
                overtaking_labs = []
                if new_rank > base_result['rank']:
                    for r in new_ranking[:new_rank - 1]:
                        # 基準ランキングでの順位を確認
                        base_r = self.matcher.find_lab_result(base_ranking, r['lab_name'])
                        if base_r and base_r['rank'] > base_result['rank']:
                            overtaking_labs.append({
                                'lab_name': r['lab_name'],
                                'new_rank': r['rank'],
                                'base_rank': base_r['rank']
                            })
                
                # 結果を記録
                param_result['test_results'][test_value] = {
                    'value': test_value,
                    'rank': new_rank,
                    'rank_change': rank_change,
                    'final_score': round(new_score, 6),
                    'basic_score': round(new_basic, 6),
                    'overtaking_labs': overtaking_labs[:5]
                }
                
                param_result['ranks'].append(new_rank)
                param_result['scores'].append(round(new_score, 6))
                param_result['basic_scores'].append(round(new_basic, 6))
                
                # 最小・最大順位を更新
                if new_rank < param_result['min_rank']:
                    param_result['min_rank'] = new_rank
                    param_result['best_value'] = test_value
                if new_rank > param_result['max_rank']:
                    param_result['max_rank'] = new_rank
                    param_result['worst_value'] = test_value
            
            # 変動幅を計算
            param_result['rank_range'] = param_result['max_rank'] - param_result['min_rank']
            param_result['has_variation'] = param_result['rank_range'] > 0
            
            results['parameters'][param] = param_result
            
            # サマリー更新
            if param_result['has_variation']:
                results['summary']['params_with_variation'] += 1
                if param_result['rank_range'] > results['summary']['max_rank_drop']:
                    results['summary']['max_rank_drop'] = param_result['rank_range']
                    results['summary']['most_sensitive_param'] = param
        
        return results
    
    def analyze_all_labs(self) -> Dict:
        """全研究室の感度分析"""
        all_results = {}
        
        for i, optimal_data in enumerate(self.optimal_solutions, 1):
            lab_name = optimal_data.get('lab_name', '')
            print(f"[{i}/{len(self.optimal_solutions)}] {lab_name}", end="", flush=True)
            
            result = self.analyze_single_lab(optimal_data)
            all_results[lab_name] = result
            
            if 'error' not in result:
                variation_count = result['summary']['params_with_variation']
                max_drop = result['summary']['max_rank_drop']
                if variation_count > 0:
                    print(f" → {variation_count}項目で変動 (最大{max_drop}位)")
                else:
                    print(" → 変動なし")
            else:
                print(f" → エラー: {result['error']}")
        
        return all_results


# =============================================================================
# レポート生成
# =============================================================================

def generate_full_report(all_results: Dict) -> str:
    """全データを含む詳細レポート生成"""
    lines = []
    
    # ヘッダー
    lines.append("=" * 120)
    lines.append("完全版 感度分析レポート（全データ記録）")
    lines.append("最適解を軸に11項目すべてを1-10で変動")
    lines.append("=" * 120)
    lines.append(f"分析日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"分析対象: {len(all_results)}ゼミ")
    lines.append(f"分析項目: {len(BASIC_CRITERIA)}項目")
    lines.append(f"テスト値: {TEST_VALUES}")
    lines.append(f"σ値: {SIGMA}")
    lines.append(f"総データポイント: {len(all_results)} × {len(BASIC_CRITERIA)} × {len(TEST_VALUES)} = {len(all_results) * len(BASIC_CRITERIA) * len(TEST_VALUES)}")
    lines.append("")
    
    # 全体統計
    param_ranges = defaultdict(list)
    overtaker_counts = defaultdict(int)
    
    for lab_name, data in all_results.items():
        if 'error' in data:
            continue
        for param, pdata in data.get('parameters', {}).items():
            param_ranges[param].append(pdata['rank_range'])
            # 上位に来たゼミをカウント
            for v, vdata in pdata.get('test_results', {}).items():
                for ot in vdata.get('overtaking_labs', []):
                    overtaker_counts[ot['lab_name']] += 1
    
    # パラメータ別サマリー
    lines.append("\n" + "=" * 100)
    lines.append("【パラメータ別影響度サマリー】")
    lines.append("=" * 100)
    lines.append(f"{'パラメータ':<20} {'平均変動幅':>12} {'最大変動幅':>12} {'変動ゼミ数':>12}")
    lines.append("-" * 60)
    
    param_summary = []
    for param in BASIC_CRITERIA:
        ranges = param_ranges.get(param, [])
        if ranges:
            avg = sum(ranges) / len(ranges)
            max_val = max(ranges)
            count = sum(1 for r in ranges if r > 0)
            param_summary.append((param, avg, max_val, count))
    
    param_summary.sort(key=lambda x: (x[3], x[2]), reverse=True)
    
    for param, avg, max_val, count in param_summary:
        ja = PARAM_NAMES_JA.get(param, param)
        lines.append(f"{ja:<20} {avg:>12.2f} {max_val:>12} {count:>12}")
    
    # 上位に来やすいゼミ
    if overtaker_counts:
        lines.append(f"\n■ パラメータ変動時に上位に来たゼミ（回数順）")
        lines.append("-" * 60)
        sorted_overtakers = sorted(overtaker_counts.items(), key=lambda x: -x[1])[:15]
        for lab, count in sorted_overtakers:
            lines.append(f"  {lab}: {count}回")
    
    # 各ゼミ詳細（全データ）
    lines.append("\n" + "=" * 120)
    lines.append("【個別ゼミ詳細（全11項目×10値データ）】")
    lines.append("=" * 120)
    
    for lab_name in sorted(all_results.keys()):
        data = all_results[lab_name]
        
        if 'error' in data:
            lines.append(f"\n{lab_name}: エラー - {data['error']}")
            continue
        
        lines.append(f"\n{'='*100}")
        lines.append(f"【{lab_name}】")
        lines.append(f"  基準順位: {data['base_rank']}位")
        lines.append(f"  基準スコア: {data['base_score']} (基本: {data['base_basic_score']}, 分野: {data['base_field_score']})")
        lines.append(f"  対象分野: {data['lab_field_id']}")
        lines.append(f"  変動項目数: {data['summary']['params_with_variation']}/{data['summary']['total_params']}")
        
        if data['summary']['most_sensitive_param']:
            sens_param = PARAM_NAMES_JA.get(data['summary']['most_sensitive_param'], data['summary']['most_sensitive_param'])
            lines.append(f"  最敏感項目: {sens_param} (最大{data['summary']['max_rank_drop']}位変動)")
        
        # 全11項目のデータを表示
        for param in BASIC_CRITERIA:
            pdata = data['parameters'].get(param)
            if not pdata:
                continue
            
            ja = pdata['param_name_ja']
            has_change = pdata['has_variation']
            change_mark = "★変動あり" if has_change else ""
            
            lines.append(f"\n  ● {ja} {change_mark}")
            lines.append(f"    最適値: {pdata['original_value']} (重要度: {pdata['original_priority']})")
            lines.append(f"    研究室値: {pdata['lab_value']}")
            lines.append(f"    順位範囲: {pdata['min_rank']}位 ～ {pdata['max_rank']}位 (変動幅: {pdata['rank_range']})")
            
            if pdata['best_value'] != pdata['worst_value']:
                lines.append(f"    最良値: {pdata['best_value']} → {pdata['min_rank']}位")
                lines.append(f"    最悪値: {pdata['worst_value']} → {pdata['max_rank']}位")
            
            # 値ごとのデータを表形式で
            lines.append(f"    {'値:':<8}" + " ".join([f"{v:>6}" for v in TEST_VALUES]))
            lines.append(f"    {'順位:':<8}" + " ".join([f"{r:>6}" for r in pdata['ranks']]))
            lines.append(f"    {'スコア:':<8}" + " ".join([f"{s:>6.4f}" for s in pdata['scores']]))
            lines.append(f"    {'基本:':<8}" + " ".join([f"{s:>6.4f}" for s in pdata['basic_scores']]))
            
            # 変動があった場合の詳細
            if has_change:
                for v in TEST_VALUES:
                    vdata = pdata['test_results'].get(v, {})
                    if vdata.get('rank_change', 0) > 0 and vdata.get('overtaking_labs'):
                        overtakers = ", ".join([ot['lab_name'] for ot in vdata['overtaking_labs'][:3]])
                        lines.append(f"      値{v}: {vdata['rank_change']:+d}位 → 上位: {overtakers}")
    
    return "\n".join(lines)


def generate_csv_data(all_results: Dict) -> str:
    """CSV形式のデータを生成（Excel対応）"""
    lines = []
    
    # ヘッダー
    header_parts = [
        "ゼミ名", "ゼミID", "分野ID", "基準順位", "基準スコア",
        "パラメータ", "パラメータ(日本語)", "最適値", "重要度", "研究室値", "変動幅"
    ]
    header_parts += [f"値{v}_順位" for v in TEST_VALUES]
    header_parts += [f"値{v}_スコア" for v in TEST_VALUES]
    header_parts += [f"値{v}_基本スコア" for v in TEST_VALUES]
    lines.append(",".join(header_parts))
    
    # データ行
    for lab_name in sorted(all_results.keys()):
        data = all_results[lab_name]
        
        if 'error' in data:
            continue
        
        for param in BASIC_CRITERIA:
            pdata = data['parameters'].get(param)
            if not pdata:
                continue
            
            row = [
                lab_name,
                data.get('lab_id', ''),
                data.get('lab_field_id', ''),
                str(data['base_rank']),
                str(data['base_score']),
                param,
                pdata['param_name_ja'],
                str(pdata['original_value']),
                str(pdata['original_priority']),
                str(pdata['lab_value']),
                str(pdata['rank_range'])
            ]
            row += [str(r) for r in pdata['ranks']]
            row += [str(s) for s in pdata['scores']]
            row += [str(s) for s in pdata['basic_scores']]
            
            lines.append(",".join(row))
    
    return "\n".join(lines)


# =============================================================================
# データ読み込み
# =============================================================================

def load_data() -> Tuple[List[Dict], List[Dict]]:
    """データファイルを読み込み"""
    
    optimal_paths = [
        Path('results/genetic_optimization/all_labs_summary.json'),
        Path('backend/results/genetic_optimization/all_labs_summary.json'),
    ]
    
    labs_paths = [
        Path('data/labs_database.json'),
        Path('backend/data/labs_database.json'),
    ]
    
    optimal_path = None
    labs_path = None
    
    for p in optimal_paths:
        if p.exists():
            optimal_path = p
            break
    
    for p in labs_paths:
        if p.exists():
            labs_path = p
            break
    
    if not optimal_path or not labs_path:
        print("❌ データファイルが見つかりません")
        print(f"  試行した最適解パス: {optimal_paths}")
        print(f"  試行した研究室パス: {labs_paths}")
        sys.exit(1)
    
    print(f"  最適解: {optimal_path}")
    print(f"  研究室: {labs_path}")
    
    with open(optimal_path, 'r', encoding='utf-8') as f:
        optimal_data = json.load(f)
    
    with open(labs_path, 'r', encoding='utf-8') as f:
        labs_data = json.load(f)
    
    # データ構造に対応
    if isinstance(labs_data, dict):
        labs = labs_data.get('labs', [])
    else:
        labs = labs_data
    
    if isinstance(optimal_data, dict):
        optimal_solutions = optimal_data.get('results', [])
    else:
        optimal_solutions = optimal_data
    
    return labs, optimal_solutions


# =============================================================================
# メイン
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="完全版感度分析（全データ記録）")
    parser.add_argument('--mode', choices=['full', 'debug'], default='full',
                        help='実行モード: full=全分析, debug=デバッグ')
    parser.add_argument('--no-confirm', action='store_true',
                        help='確認なしで実行')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("完全版 感度分析（全データ記録）")
    print("11項目 × 10値 = 110データポイント/ゼミ")
    print("=" * 80)
    
    # データ読み込み
    labs, optimal_solutions = load_data()
    print(f"  研究室: {len(labs)}件")
    print(f"  最適解: {len(optimal_solutions)}件")
    print(f"  総データポイント: {len(optimal_solutions) * len(BASIC_CRITERIA) * len(TEST_VALUES)}")
    
    # 分析器作成
    analyzer = CompleteSensitivityAnalyzer(labs, optimal_solutions)
    
    if args.mode == 'debug':
        # デバッグモード：最初のゼミのみ詳細表示
        if optimal_solutions:
            opt = optimal_solutions[0]
            lab_name = opt.get('lab_name', '')
            
            print(f"\n【デバッグモード】{lab_name}")
            
            result = analyzer.analyze_single_lab(opt)
            
            if 'error' not in result:
                print(f"基準順位: {result['base_rank']}位")
                print(f"基準スコア: {result['base_score']}")
                
                print(f"\n全11項目のデータ:")
                for param in BASIC_CRITERIA:
                    pdata = result['parameters'].get(param)
                    if pdata:
                        mark = "★" if pdata['has_variation'] else ""
                        print(f"  {pdata['param_name_ja']:<16} 最適値:{pdata['original_value']:>6.2f} 変動:{pdata['rank_range']}{mark}")
                        print(f"    順位: {pdata['ranks']}")
        return 0
    
    # 全ゼミ分析
    if not args.no_confirm:
        print(f"\n全{len(optimal_solutions)}ゼミの分析を開始します。")
        print(f"（{len(optimal_solutions) * len(BASIC_CRITERIA) * len(TEST_VALUES)}回の計算）")
        input("準備ができたらEnterキーを押してください...")
    
    print("\n分析開始...")
    all_results = analyzer.analyze_all_labs()
    
    # レポート生成
    print("\nレポート生成中...")
    report = generate_full_report(all_results)
    csv_data = generate_csv_data(all_results)
    
    # ファイル保存
    with open('sensitivity_all_data_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    with open('sensitivity_all_data.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    with open('sensitivity_all_data.csv', 'w', encoding='utf-8-sig') as f:
        f.write(csv_data)
    
    print("\n" + "=" * 80)
    print("保存完了:")
    print("  - sensitivity_all_data_report.txt  (詳細レポート)")
    print("  - sensitivity_all_data.json        (JSON形式・全データ)")
    print("  - sensitivity_all_data.csv         (CSV形式・Excel対応)")
    print("=" * 80)
    
    # レポート出力
    print("\n" + report)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())