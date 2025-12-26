#!/usr/bin/env python3
"""
感度分析 v7 Final - 柔軟なデータ構造に対応
遺伝的アルゴリズムの最適解から感度分析を実行

【特徴】
1. σ値を引数で指定可能（デフォルト: 0.2）
2. カスタムfield_idに対応
3. 複数のデータパスを自動検索
4. デバッグモード付き

使い方:
  python sensitivity_analysis_v7_final.py --mode full --sigma 0.2
  python sensitivity_analysis_v7_final.py --mode full --sigma 0.3
  python sensitivity_analysis_v7_final.py --mode debug
"""

import json
import math
import sys
from typing import Dict, List, Tuple, Any
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# =============================================================================
# 定数定義
# =============================================================================

BASIC_CRITERIA = [
    "research_intensity", "advisor_style", "team_work", "workload",
    "theory_practice", "skill_development", "lab_atmosphere", "flexibility",
    "publication_opportunity", "interdisciplinary", "communication_style"
]

PARAM_NAMES_JA = {
    'research_intensity': '研究強度', 'advisor_style': '指導スタイル',
    'team_work': 'チームワーク', 'workload': 'ワークロード',
    'theory_practice': '理論/実践', 'skill_development': 'スキル開発',
    'lab_atmosphere': '研究室雰囲気', 'flexibility': '柔軟性',
    'publication_opportunity': '発表機会', 'interdisciplinary': '学際性',
    'communication_style': 'コミュニケーション'
}

# 標準20分野
FIELD_NAMES = [
    'ai_ml', 'image_processing', 'network_security', 'database_systems',
    'embedded_iot', 'education_linguistics', 'natural_science_math',
    'tourism_regional', 'business_decision', 'audio_processing',
    'system_ethics', 'medical_healthcare', 'web_design', 'design_visual',
    'video_animation', 'computer_music', 'game_esports', 'vr_ar_media',
    'philosophy_humanities', 'sports_science'
]

# カテゴリマッピング（カスタムfield_idも追加）
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
    "エンターテイメント": ["game_esports", "vr_ar_media", "game_dev", "media_art"],
    "人文・社会・体育": [
        "philosophy_humanities", "sports_science", "english_humanities",
        "japanese_education", "korean_studies", "natural_science"
    ]
}

def get_field_category(field_id: str) -> str:
    """分野IDからカテゴリを取得（カスタムfield_idにも対応）"""
    for category, fields in FIELD_CATEGORIES.items():
        if field_id in fields:
            return category
    # 不明な場合はキーワードから推測
    if any(kw in field_id.lower() for kw in ['design', 'visual', 'art', 'graphic', 'illustration']):
        return "クリエイティブ"
    if any(kw in field_id.lower() for kw in ['game', 'vr', 'ar', 'media']):
        return "エンターテイメント"
    if any(kw in field_id.lower() for kw in ['human', 'education', 'language', 'sport', 'science']):
        return "人文・社会・体育"
    return "テクノロジー・システム"  # デフォルト


class SensitivityAnalyzer:
    """感度分析クラス"""
    
    def __init__(self, labs_data: List[Dict], optimal_solutions: List[Dict], sigma: float = 0.2):
        self.labs_data = labs_data
        self.optimal_solutions = optimal_solutions
        self.sigma = sigma
        
        # 研究室のfield_idを抽出
        self.lab_field_ids = {}
        for lab in labs_data:
            lab_name = lab.get('name') or lab.get('lab_name', '')
            self.lab_field_ids[lab_name] = self._get_lab_field_id(lab)
    
    def _get_lab_field_id(self, lab: Dict) -> str:
        """研究室の専門分野IDを取得"""
        if 'field_id' in lab:
            return lab['field_id']
        
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
            'システム': 'system_ethics', '医療': 'medical_healthcare',
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
        return 'system_ethics'
    
    def gaussian_similarity(self, v1: float, v2: float) -> float:
        """ガウス類似度計算"""
        v1_norm = (v1 - 1) / 9
        v2_norm = (v2 - 1) / 9
        d = abs(v1_norm - v2_norm)
        return math.exp(-(d ** 2) / (2 * self.sigma ** 2))
    
    def calculate_basic_score(self, student_profile: Dict, lab_profile: Dict) -> float:
        """基本項目スコアを計算"""
        total_score = 0
        total_weight = 0
        
        for criterion in BASIC_CRITERIA:
            student_val = float(student_profile.get(criterion, 5.5))
            lab_val = float(lab_profile.get(criterion, 5.5))
            priority = float(student_profile.get(f"{criterion}_priority", 5.0))
            
            sim = self.gaussian_similarity(student_val, lab_val)
            total_score += sim * priority
            total_weight += priority
        
        return total_score / total_weight if total_weight > 0 else 0
    
    def calculate_field_score(self, student_field_interests: Dict, lab_field_id: str) -> float:
        """分野スコアを計算"""
        if not lab_field_id or not student_field_interests:
            return 0.3
        
        lab_category = get_field_category(lab_field_id)
        best_score = 0.3
        
        for field_id, interest in student_field_interests.items():
            interest_norm = float(interest) / 10
            
            if field_id == lab_field_id:
                score = interest_norm
            elif get_field_category(field_id) == lab_category:
                score = interest_norm * 0.7
            else:
                score = 0.3
            
            best_score = max(best_score, score)
        
        return best_score
    
    def calculate_final_score(self, student_profile: Dict, lab_profile: Dict, 
                              student_field_interests: Dict, lab_field_id: str) -> float:
        """最終スコアを計算"""
        research_field_match = float(student_profile.get('research_field_match', 5))
        alpha = research_field_match / 10
        beta = 1 - alpha
        
        basic_score = self.calculate_basic_score(student_profile, lab_profile)
        field_score = self.calculate_field_score(student_field_interests, lab_field_id)
        
        return beta * basic_score + alpha * field_score
    
    def get_all_rankings(self, student_profile: Dict, field_interests: Dict) -> List[Dict]:
        """全研究室のスコアと順位を計算"""
        results = []
        
        for lab in self.labs_data:
            lab_name = lab.get('name') or lab.get('lab_name', '')
            lab_field_id = self.lab_field_ids.get(lab_name, 'system_ethics')
            
            score = self.calculate_final_score(student_profile, lab, field_interests, lab_field_id)
            
            results.append({
                'lab_name': lab_name,
                'field_id': lab_field_id,
                'score': score
            })
        
        results.sort(key=lambda x: -x['score'])
        for i, r in enumerate(results):
            r['rank'] = i + 1
        
        return results
    
    def find_lab_info(self, ranking: List[Dict], target_lab: str) -> Tuple[int, float]:
        """指定研究室の順位とスコアを取得"""
        for r in ranking:
            if target_lab in r['lab_name'] or r['lab_name'] in target_lab:
                return r['rank'], r['score']
        return -1, 0.0
    
    def get_field_interests(self, optimal_data: Dict) -> Dict:
        """最適解データからfield_interestsを取得"""
        profile = optimal_data.get('optimal_student_profile', {})
        
        if 'field_interests' in profile:
            fi = profile['field_interests']
            if isinstance(fi, dict):
                return fi
        
        if 'field_interests' in optimal_data:
            fi = optimal_data['field_interests']
            if isinstance(fi, dict):
                return fi
        
        # デフォルト
        lab_name = optimal_data.get('lab_name', '')
        lab_field_id = optimal_data.get('lab_field_id', self.lab_field_ids.get(lab_name, 'system_ethics'))
        return {lab_field_id: 8.0}
    
    def analyze_single_lab(self, target_lab_name: str, optimal_profile: Dict, 
                           field_interests: Dict, deltas: List[int] = [1, 2, 3]) -> Dict:
        """単一ゼミの感度分析"""
        
        base_ranking = self.get_all_rankings(optimal_profile, field_interests)
        base_rank, base_score = self.find_lab_info(base_ranking, target_lab_name)
        
        if base_rank == -1:
            return {'error': f'{target_lab_name} not found'}
        
        results = {
            'base_rank': base_rank,
            'base_score': round(base_score, 4),
            'base_top5': [(r['lab_name'], r['rank'], round(r['score'], 4)) for r in base_ranking[:5]],
            'parameters': {}
        }
        
        for param in BASIC_CRITERIA:
            if param not in optimal_profile:
                continue
            
            original_value = float(optimal_profile[param])
            param_result = {
                'original_value': round(original_value, 2),
                'variations': [],
                'max_change': 0
            }
            
            for delta in deltas:
                # +delta
                modified_plus = optimal_profile.copy()
                new_val_plus = min(10.0, original_value + delta)
                modified_plus[param] = new_val_plus
                
                ranking_plus = self.get_all_rankings(modified_plus, field_interests)
                rank_plus, score_plus = self.find_lab_info(ranking_plus, target_lab_name)
                
                overtaking_plus = []
                if rank_plus > base_rank:
                    for r in ranking_plus[:rank_plus-1]:
                        base_r, _ = self.find_lab_info(base_ranking, r['lab_name'])
                        if base_r > base_rank:
                            overtaking_plus.append((r['lab_name'], r['rank']))
                
                # -delta
                modified_minus = optimal_profile.copy()
                new_val_minus = max(1.0, original_value - delta)
                modified_minus[param] = new_val_minus
                
                ranking_minus = self.get_all_rankings(modified_minus, field_interests)
                rank_minus, score_minus = self.find_lab_info(ranking_minus, target_lab_name)
                
                overtaking_minus = []
                if rank_minus > base_rank:
                    for r in ranking_minus[:rank_minus-1]:
                        base_r, _ = self.find_lab_info(base_ranking, r['lab_name'])
                        if base_r > base_rank:
                            overtaking_minus.append((r['lab_name'], r['rank']))
                
                change_plus = rank_plus - base_rank
                change_minus = rank_minus - base_rank
                
                param_result['variations'].append({
                    'delta': delta,
                    'plus': {
                        'new_value': round(new_val_plus, 2),
                        'new_rank': rank_plus,
                        'rank_change': change_plus,
                        'overtaking_labs': overtaking_plus[:5]
                    },
                    'minus': {
                        'new_value': round(new_val_minus, 2),
                        'new_rank': rank_minus,
                        'rank_change': change_minus,
                        'overtaking_labs': overtaking_minus[:5]
                    }
                })
                
                param_result['max_change'] = max(
                    param_result['max_change'],
                    abs(change_plus),
                    abs(change_minus)
                )
            
            results['parameters'][param] = param_result
        
        return results
    
    def analyze_all(self) -> Dict:
        """全研究室の分析"""
        all_results = {}
        
        for i, optimal_data in enumerate(self.optimal_solutions, 1):
            lab_name = optimal_data.get('lab_name', '')
            optimal_profile = optimal_data.get('optimal_student_profile', {})
            field_interests = self.get_field_interests(optimal_data)
            
            print(f"\n[{i}/{len(self.optimal_solutions)}] {lab_name}")
            
            if not optimal_profile:
                print(f"  ⚠️ optimal_student_profile がありません")
                continue
            
            results = self.analyze_single_lab(lab_name, optimal_profile, field_interests)
            
            if 'error' not in results:
                print(f"  基準順位: {results['base_rank']}位")
                
                changes = [(p, d['max_change']) for p, d in results['parameters'].items() if d['max_change'] > 0]
                if changes:
                    changes.sort(key=lambda x: x[1], reverse=True)
                    for p, c in changes[:3]:
                        print(f"  - {PARAM_NAMES_JA.get(p, p)}: ±{c}位")
                else:
                    print("  - 変動なし")
            
            all_results[lab_name] = results
        
        return all_results


def generate_report(all_results: Dict, sigma: float) -> str:
    """レポート生成"""
    lines = []
    lines.append("=" * 100)
    lines.append(f"感度分析レポート v7 Final (σ={sigma})")
    lines.append("=" * 100)
    lines.append(f"分析日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"分析対象: {len(all_results)}ゼミ")
    lines.append("")
    
    param_max_changes = defaultdict(list)
    overtakers = defaultdict(int)
    
    for lab_name, data in all_results.items():
        if 'error' in data:
            continue
        
        for param, pdata in data.get('parameters', {}).items():
            param_max_changes[param].append(pdata['max_change'])
            
            for v in pdata.get('variations', []):
                for ot in v['plus'].get('overtaking_labs', []):
                    overtakers[ot[0]] += 1
                for ot in v['minus'].get('overtaking_labs', []):
                    overtakers[ot[0]] += 1
    
    lines.append("\n" + "=" * 80)
    lines.append("【全体サマリー】パラメータ別影響度")
    lines.append("=" * 80)
    lines.append(f"{'パラメータ':<20} {'平均変動':>10} {'最大変動':>10} {'変動ゼミ数':>12}")
    lines.append("-" * 55)
    
    param_summary = []
    for param in BASIC_CRITERIA:
        changes = param_max_changes.get(param, [])
        if changes:
            avg = sum(changes) / len(changes)
            max_val = max(changes)
            count = sum(1 for c in changes if c > 0)
            param_summary.append((param, avg, max_val, count))
    
    param_summary.sort(key=lambda x: (x[3], x[1]), reverse=True)
    
    for param, avg, max_val, count in param_summary:
        ja_name = PARAM_NAMES_JA.get(param, param)
        lines.append(f"{ja_name:<20} {avg:>10.2f} {max_val:>10} {count:>12}")
    
    if overtakers:
        lines.append(f"\n■ 他ゼミのパラメータ変動時に上位に来たゼミ")
        lines.append("-" * 60)
        sorted_overtakers = sorted(overtakers.items(), key=lambda x: x[1], reverse=True)[:15]
        for lab, count in sorted_overtakers:
            lines.append(f"  {lab}: {count}回")
    
    lines.append("\n" + "=" * 100)
    lines.append("【個別ゼミ詳細】")
    lines.append("=" * 100)
    
    for lab_name, data in sorted(all_results.items()):
        if 'error' in data:
            continue
        
        lines.append(f"\n{'='*70}")
        lines.append(f"【{lab_name}】")
        lines.append(f"  基準順位: {data.get('base_rank')}位 (スコア: {data.get('base_score')})")
        
        has_change = False
        for param, pdata in data.get('parameters', {}).items():
            if pdata['max_change'] == 0:
                continue
            
            has_change = True
            ja_name = PARAM_NAMES_JA.get(param, param)
            lines.append(f"\n  ● {ja_name} (元値: {pdata['original_value']}, 最大変動: ±{pdata['max_change']})")
            
            for v in pdata.get('variations', []):
                delta = v['delta']
                
                if v['plus']['rank_change'] != 0:
                    direction = "↓" if v['plus']['rank_change'] > 0 else "↑"
                    lines.append(f"    +{delta} → {v['plus']['new_rank']}位 ({direction}{abs(v['plus']['rank_change'])})")
                    for ot in v['plus'].get('overtaking_labs', [])[:3]:
                        lines.append(f"       ▲上位: {ot[0]} ({ot[1]}位)")
                
                if v['minus']['rank_change'] != 0:
                    direction = "↓" if v['minus']['rank_change'] > 0 else "↑"
                    lines.append(f"    -{delta} → {v['minus']['new_rank']}位 ({direction}{abs(v['minus']['rank_change'])})")
                    for ot in v['minus'].get('overtaking_labs', [])[:3]:
                        lines.append(f"       ▲上位: {ot[0]} ({ot[1]}位)")
        
        if not has_change:
            lines.append("  ※ 全パラメータで順位変動なし")
    
    return "\n".join(lines)


def load_data():
    """データ読み込み"""
    optimal_paths = [
        Path('results/genetic_optimization/all_labs_summary.json'),
        Path('backend/results/genetic_optimization/all_labs_summary.json'),
        Path('results/improved_ga/all_labs_priority_rfm5.json'),
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
        print(f"❌ データファイルが見つかりません")
        sys.exit(1)
    
    print(f"データ読み込み中...")
    print(f"  最適解: {optimal_path}")
    print(f"  研究室: {labs_path}")
    
    with open(optimal_path, 'r', encoding='utf-8') as f:
        optimal_data = json.load(f)
    
    with open(labs_path, 'r', encoding='utf-8') as f:
        labs_data = json.load(f)
    
    if isinstance(labs_data, dict):
        labs = labs_data.get('labs', [])
    else:
        labs = labs_data
    
    # フラット化
    flattened_labs = []
    for lab in labs:
        flat_lab = lab.copy()
        if 'features' in lab:
            for key, value in lab['features'].items():
                flat_lab[key] = value
        flattened_labs.append(flat_lab)
    
    if isinstance(optimal_data, dict):
        optimal_solutions = optimal_data.get('results', [])
    else:
        optimal_solutions = optimal_data
    
    return flattened_labs, optimal_solutions


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="感度分析 v7 Final")
    parser.add_argument('--mode', choices=['full', 'debug'], default='full')
    parser.add_argument('--sigma', type=float, default=0.2, help='ガウス類似度のσ値')
    parser.add_argument('--no-confirm', action='store_true')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print(f"感度分析 v7 Final (σ={args.sigma})")
    print("=" * 70)
    
    labs, optimal_solutions = load_data()
    
    print(f"  研究室データ: {len(labs)}件")
    print(f"  最適解データ: {len(optimal_solutions)}件")
    
    analyzer = SensitivityAnalyzer(labs, optimal_solutions, sigma=args.sigma)
    
    if args.mode == 'debug':
        if optimal_solutions:
            sample = optimal_solutions[0]
            lab_name = sample.get('lab_name', '')
            profile = sample.get('optimal_student_profile', {})
            field_interests = analyzer.get_field_interests(sample)
            
            print(f"\n【詳細デバッグ】{lab_name}")
            print(f"  field_interests: {field_interests}")
            print(f"  lab_field_id: {analyzer.lab_field_ids.get(lab_name)}")
            
            base_ranking = analyzer.get_all_rankings(profile, field_interests)
            print(f"\n  基準ランキング TOP10:")
            for r in base_ranking[:10]:
                print(f"    {r['rank']}位: {r['lab_name']} ({r['field_id']}) スコア:{r['score']:.4f}")
        
        return 0
    
    all_results = analyzer.analyze_all()
    
    print("\n\nレポート生成中...")
    report = generate_report(all_results, args.sigma)
    
    with open(f'sensitivity_report_v7_sigma{args.sigma}.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    with open(f'sensitivity_v7_sigma{args.sigma}.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 70)
    print("保存完了:")
    print(f"  - sensitivity_report_v7_sigma{args.sigma}.txt")
    print(f"  - sensitivity_v7_sigma{args.sigma}.json")
    print("=" * 70)
    
    print("\n" + report)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())