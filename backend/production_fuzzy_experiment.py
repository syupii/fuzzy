#!/usr/bin/env python3
"""
モンテカルロ法による公平性検証実験
=============================================
プロダクションコード（fuzzy_multipath_matcher.py）完全準拠版

実際のシステムと同一のパラメータ・アルゴリズムを使用:
- SIMILARITY_SIGMA = 0.3（実際のコード準拠）
- 優先度の非線形変換（priority^1.5）
- 分野ボーナス/ペナルティシステム
- 技術資料3.4.1節のメンバーシップ関数
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
import json
import time
from datetime import datetime
import math


# ============================================================
# 北海道情報大学 31研究室データベース（実データ）
# ============================================================

LABS_DATABASE = [
    {"id": "lab_001", "name": "坂本 ゼミ", "professor": "坂本 牧葉", "field_id": "graphic_visual", "research_area": "視覚デザイン",
     "features": {"research_intensity": 5, "advisor_style": 6, "team_work": 7, "workload": 5, "theory_practice": 9, "skill_development": 6, "lab_atmosphere": 8, "flexibility": 7, "publication_opportunity": 4, "interdisciplinary": 8, "communication_style": 8}},
    {"id": "lab_002", "name": "伊藤 正彦ゼミ", "professor": "伊藤 正彦", "field_id": "database_systems", "research_area": "情報可視化",
     "features": {"research_intensity": 9, "advisor_style": 8, "team_work": 5, "workload": 7, "theory_practice": 8, "skill_development": 7, "lab_atmosphere": 9, "flexibility": 9, "publication_opportunity": 9, "interdisciplinary": 8, "communication_style": 9}},
    {"id": "lab_003", "name": "斎藤 一 ゼミ", "professor": "斎藤 一", "field_id": "tourism_regional", "research_area": "観光情報学",
     "features": {"research_intensity": 6, "advisor_style": 7, "team_work": 8, "workload": 6, "theory_practice": 10, "skill_development": 9, "lab_atmosphere": 9, "flexibility": 7, "publication_opportunity": 5, "interdisciplinary": 9, "communication_style": 9}},
    {"id": "lab_004", "name": "広奥ゼミ", "professor": "広奥", "field_id": "network_security", "research_area": "セキュリティ",
     "features": {"research_intensity": 8, "advisor_style": 7, "team_work": 3, "workload": 9, "theory_practice": 6, "skill_development": 9, "lab_atmosphere": 5, "flexibility": 7, "publication_opportunity": 6, "interdisciplinary": 4, "communication_style": 6}},
    {"id": "lab_005", "name": "飯嶋 ゼミ", "professor": "飯嶋 美知子", "field_id": "japanese_education", "research_area": "日本語教育学",
     "features": {"research_intensity": 4, "advisor_style": 8, "team_work": 3, "workload": 3, "theory_practice": 2, "skill_development": 3, "lab_atmosphere": 8, "flexibility": 9, "publication_opportunity": 3, "interdisciplinary": 6, "communication_style": 7}},
    {"id": "lab_006", "name": "向田ゼミ", "professor": "向田", "field_id": "image_processing", "research_area": "画像処理",
     "features": {"research_intensity": 8, "advisor_style": 6, "team_work": 7, "workload": 7, "theory_practice": 10, "skill_development": 9, "lab_atmosphere": 7, "flexibility": 3, "publication_opportunity": 8, "interdisciplinary": 7, "communication_style": 7}},
    {"id": "lab_007", "name": "安田ゼミ", "professor": "安田", "field_id": "web_design_uiux", "research_area": "Webデザイン",
     "features": {"research_intensity": 7, "advisor_style": 6, "team_work": 9, "workload": 7, "theory_practice": 10, "skill_development": 8, "lab_atmosphere": 9, "flexibility": 5, "publication_opportunity": 5, "interdisciplinary": 8, "communication_style": 9}},
    {"id": "lab_008", "name": "森川ゼミ", "professor": "森川", "field_id": "game_dev", "research_area": "ゲームプログラミング",
     "features": {"research_intensity": 9, "advisor_style": 7, "team_work": 7, "workload": 8, "theory_practice": 10, "skill_development": 7, "lab_atmosphere": 9, "flexibility": 6, "publication_opportunity": 10, "interdisciplinary": 5, "communication_style": 8}},
    {"id": "lab_009", "name": "甫喜本ゼミ", "professor": "甫喜本 司", "field_id": "data_science_math", "research_area": "データ解析法",
     "features": {"research_intensity": 7, "advisor_style": 7, "team_work": 2, "workload": 6, "theory_practice": 3, "skill_development": 4, "lab_atmosphere": 4, "flexibility": 7, "publication_opportunity": 3, "interdisciplinary": 6, "communication_style": 4}},
    {"id": "lab_010", "name": "金ゼミ", "professor": "金 銀珠", "field_id": "korean_studies", "research_area": "韓国IT・コンテンツ産業研究",
     "features": {"research_intensity": 5, "advisor_style": 6, "team_work": 4, "workload": 5, "theory_practice": 5, "skill_development": 5, "lab_atmosphere": 8, "flexibility": 7, "publication_opportunity": 3, "interdisciplinary": 6, "communication_style": 8}},
    {"id": "lab_011", "name": "辻ゼミ", "professor": "辻 順平", "field_id": "ai_ml", "research_area": "人工知能",
     "features": {"research_intensity": 8, "advisor_style": 7, "team_work": 7, "workload": 8, "theory_practice": 7, "skill_development": 8, "lab_atmosphere": 9, "flexibility": 6, "publication_opportunity": 9, "interdisciplinary": 7, "communication_style": 8}},
    {"id": "lab_012", "name": "齋藤健司ゼミ", "professor": "齋藤健司", "field_id": "ai_ml", "research_area": "人工知能",
     "features": {"research_intensity": 6, "advisor_style": 7, "team_work": 5, "workload": 6, "theory_practice": 6, "skill_development": 7, "lab_atmosphere": 6, "flexibility": 8, "publication_opportunity": 4, "interdisciplinary": 6, "communication_style": 5}},
    {"id": "lab_013", "name": "谷口ゼミ", "professor": "谷口", "field_id": "software_dev", "research_area": "ソフトウェア開発",
     "features": {"research_intensity": 6, "advisor_style": 8, "team_work": 2, "workload": 6, "theory_practice": 10, "skill_development": 8, "lab_atmosphere": 5, "flexibility": 8, "publication_opportunity": 3, "interdisciplinary": 3, "communication_style": 4}},
    {"id": "lab_014", "name": "隼田ゼミ", "professor": "隼田", "field_id": "media_art", "research_area": "メディアアート",
     "features": {"research_intensity": 7, "advisor_style": 7, "team_work": 7, "workload": 7, "theory_practice": 8, "skill_development": 8, "lab_atmosphere": 9, "flexibility": 7, "publication_opportunity": 6, "interdisciplinary": 8, "communication_style": 9}},
    {"id": "lab_015", "name": "綿谷ゼミ", "professor": "綿谷", "field_id": "sports_science", "research_area": "スポーツバイオメカニクス",
     "features": {"research_intensity": 6, "advisor_style": 6, "team_work": 4, "workload": 5, "theory_practice": 5, "skill_development": 3, "lab_atmosphere": 7, "flexibility": 6, "publication_opportunity": 3, "interdisciplinary": 2, "communication_style": 6}},
    {"id": "lab_016", "name": "大島ゼミ", "professor": "大島", "field_id": "video_film", "research_area": "映像表現",
     "features": {"research_intensity": 7, "advisor_style": 7, "team_work": 7, "workload": 7, "theory_practice": 9, "skill_development": 7, "lab_atmosphere": 8, "flexibility": 6, "publication_opportunity": 7, "interdisciplinary": 7, "communication_style": 7}},
    {"id": "lab_017", "name": "三浦ゼミ", "professor": "三浦", "field_id": "english_humanities", "research_area": "哲学",
     "features": {"research_intensity": 4, "advisor_style": 5, "team_work": 2, "workload": 4, "theory_practice": 1, "skill_development": 2, "lab_atmosphere": 3, "flexibility": 7, "publication_opportunity": 2, "interdisciplinary": 3, "communication_style": 3}},
    {"id": "lab_018", "name": "近澤ゼミ", "professor": "近澤", "field_id": "web_design_uiux", "research_area": "Webデザイン",
     "features": {"research_intensity": 7, "advisor_style": 7, "team_work": 9, "workload": 7, "theory_practice": 9, "skill_development": 8, "lab_atmosphere": 9, "flexibility": 6, "publication_opportunity": 5, "interdisciplinary": 7, "communication_style": 9}},
    {"id": "lab_019", "name": "藤原ゼミ", "professor": "藤原", "field_id": "image_processing", "research_area": "コンピュータビジョン",
     "features": {"research_intensity": 7, "advisor_style": 7, "team_work": 5, "workload": 7, "theory_practice": 8, "skill_development": 6, "lab_atmosphere": 6, "flexibility": 7, "publication_opportunity": 4, "interdisciplinary": 5, "communication_style": 5}},
    {"id": "lab_020", "name": "山北ゼミ", "professor": "山北", "field_id": "database_systems", "research_area": "データ工学",
     "features": {"research_intensity": 6, "advisor_style": 4, "team_work": 2, "workload": 6, "theory_practice": 7, "skill_development": 4, "lab_atmosphere": 3, "flexibility": 4, "publication_opportunity": 3, "interdisciplinary": 2, "communication_style": 2}},
    {"id": "lab_021", "name": "河原ゼミ", "professor": "河原", "field_id": "game_dev", "research_area": "ゲームプログラミング",
     "features": {"research_intensity": 7, "advisor_style": 7, "team_work": 6, "workload": 7, "theory_practice": 8, "skill_development": 7, "lab_atmosphere": 8, "flexibility": 7, "publication_opportunity": 6, "interdisciplinary": 7, "communication_style": 7}},
    {"id": "lab_022", "name": "松井ゼミ", "professor": "松井", "field_id": "data_science_math", "research_area": "統計数理",
     "features": {"research_intensity": 3, "advisor_style": 6, "team_work": 2, "workload": 3, "theory_practice": 2, "skill_development": 3, "lab_atmosphere": 2, "flexibility": 8, "publication_opportunity": 2, "interdisciplinary": 1, "communication_style": 2}},
    {"id": "lab_023", "name": "平山ゼミ", "professor": "平山", "field_id": "computer_music", "research_area": "コンピュータ音楽",
     "features": {"research_intensity": 7, "advisor_style": 7, "team_work": 5, "workload": 7, "theory_practice": 7, "skill_development": 7, "lab_atmosphere": 7, "flexibility": 6, "publication_opportunity": 8, "interdisciplinary": 7, "communication_style": 7}},
    {"id": "lab_024", "name": "守 ゼミ", "professor": "守 啓祐", "field_id": "audio_processing", "research_area": "音声情報処理",
     "features": {"research_intensity": 6, "advisor_style": 7, "team_work": 4, "workload": 6, "theory_practice": 8, "skill_development": 8, "lab_atmosphere": 6, "flexibility": 7, "publication_opportunity": 3, "interdisciplinary": 6, "communication_style": 5}},
    {"id": "lab_025", "name": "伊藤マーティゼミ", "professor": "伊藤 マーティ", "field_id": "illustration_art", "research_area": "イラストレーション",
     "features": {"research_intensity": 6, "advisor_style": 7, "team_work": 7, "workload": 7, "theory_practice": 7, "skill_development": 6, "lab_atmosphere": 9, "flexibility": 7, "publication_opportunity": 6, "interdisciplinary": 7, "communication_style": 9}},
    {"id": "lab_026", "name": "柿並ゼミ", "professor": "柿並 義宏", "field_id": "natural_science", "research_area": "地球物理学",
     "features": {"research_intensity": 7, "advisor_style": 7, "team_work": 5, "workload": 7, "theory_practice": 7, "skill_development": 8, "lab_atmosphere": 7, "flexibility": 7, "publication_opportunity": 7, "interdisciplinary": 8, "communication_style": 7}},
    {"id": "lab_027", "name": "佐々木ゼミ", "professor": "佐々木 洋平", "field_id": "network_security", "research_area": "セキュリティ",
     "features": {"research_intensity": 7, "advisor_style": 8, "team_work": 7, "workload": 7, "theory_practice": 7, "skill_development": 7, "lab_atmosphere": 8, "flexibility": 8, "publication_opportunity": 5, "interdisciplinary": 8, "communication_style": 7}},
    {"id": "lab_028", "name": "新井山ゼミ", "professor": "新井山", "field_id": "software_dev", "research_area": "ソフトウェア開発",
     "features": {"research_intensity": 4, "advisor_style": 9, "team_work": 3, "workload": 3, "theory_practice": 8, "skill_development": 6, "lab_atmosphere": 7, "flexibility": 9, "publication_opportunity": 3, "interdisciplinary": 5, "communication_style": 6}},
    {"id": "lab_029", "name": "杉澤ゼミ", "professor": "杉澤 愛美", "field_id": "graphic_visual", "research_area": "グラフィックデザイン",
     "features": {"research_intensity": 7, "advisor_style": 8, "team_work": 9, "workload": 8, "theory_practice": 9, "skill_development": 8, "lab_atmosphere": 9, "flexibility": 7, "publication_opportunity": 5, "interdisciplinary": 8, "communication_style": 9}},
    {"id": "lab_030", "name": "島田ゼミ", "professor": "島田 英二", "field_id": "video_film", "research_area": "映像表現",
     "features": {"research_intensity": 8, "advisor_style": 8, "team_work": 9, "workload": 8, "theory_practice": 10, "skill_development": 8, "lab_atmosphere": 10, "flexibility": 5, "publication_opportunity": 7, "interdisciplinary": 6, "communication_style": 10}},
    {"id": "lab_031", "name": "湯村ゼミ", "professor": "湯村 翼", "field_id": "embedded_iot", "research_area": "IoT",
     "features": {"research_intensity": 9, "advisor_style": 7, "team_work": 8, "workload": 8, "theory_practice": 7, "skill_development": 9, "lab_atmosphere": 9, "flexibility": 7, "publication_opportunity": 9, "interdisciplinary": 8, "communication_style": 8}},
]


# ============================================================
# 27研究分野体系
# ============================================================

RESEARCH_FIELDS = [
    "ai_ml", "image_processing", "cg_graphics", "network_security",
    "database_systems", "embedded_iot", "software_dev", "audio_processing",
    "data_science_math", "natural_science", "japanese_education", "korean_studies",
    "educational_tech", "english_humanities", "tourism_regional", "web_design_uiux",
    "graphic_visual", "illustration_art", "design_thinking_marketing",
    "video_film", "animation", "computer_music", "media_art",
    "game_dev", "esports", "vr_ar_metaverse", "sports_science",
]

FIELD_CATEGORIES = {
    "テクノロジー・システム": [
        "ai_ml", "image_processing", "cg_graphics", "network_security",
        "database_systems", "embedded_iot", "software_dev",
        "audio_processing", "data_science_math", "natural_science"
    ],
    "教育・言語・文化": [
        "japanese_education", "korean_studies", "educational_tech", "english_humanities"
    ],
    "観光・地域": ["tourism_regional"],
    "デザイン": [
        "web_design_uiux", "graphic_visual", "illustration_art", "design_thinking_marketing"
    ],
    "映像・音楽": ["video_film", "animation", "computer_music", "media_art"],
    "ゲーム・エンタメ": ["game_dev", "esports", "vr_ar_metaverse"],
    "人文・社会・体育": ["sports_science"]
}

EVALUATION_CRITERIA = [
    "research_intensity", "advisor_style", "team_work", "workload",
    "theory_practice", "skill_development", "lab_atmosphere", "flexibility",
    "publication_opportunity", "interdisciplinary", "communication_style",
]


# ============================================================
# メンバーシップ関数（技術資料 3.4.1 完全準拠）
# ============================================================

class MembershipFunctions:
    """メンバーシップ関数（技術資料 3.4.1節の厳密な定義）"""
    
    @staticmethod
    def low(x: float) -> float:
        if x <= 0.3:
            return 1.0
        elif x < 0.5:
            return (0.5 - x) / 0.2
        else:
            return 0.0
    
    @staticmethod
    def medium(x: float) -> float:
        if x <= 0.3 or x >= 0.9:
            return 0.0
        elif x < 0.5:
            return (x - 0.3) / 0.2
        elif x <= 0.7:
            return 1.0
        else:
            return (0.9 - x) / 0.2
    
    @staticmethod
    def high(x: float) -> float:
        if x <= 0.7:
            return 0.0
        elif x < 0.9:
            return (x - 0.7) / 0.2
        else:
            return 1.0
    
    @staticmethod
    def fuzzify_3level(x: float) -> Dict[str, float]:
        return {
            "low": MembershipFunctions.low(x),
            "medium": MembershipFunctions.medium(x),
            "high": MembershipFunctions.high(x)
        }
    
    @staticmethod
    def fuzzify_2level(x: float) -> Dict[str, float]:
        if x <= 0.3:
            return {"low": 1.0, "high": 0.0}
        elif x >= 0.7:
            return {"low": 0.0, "high": 1.0}
        else:
            low_membership = max(0, (0.7 - x) / 0.4)
            high_membership = max(0, (x - 0.3) / 0.4)
            return {"low": low_membership, "high": high_membership}


# ============================================================
# ファジィパス
# ============================================================

@dataclass
class FuzzyPath:
    path_id: int
    layers: List[Tuple[str, str, float]]
    total_membership: float
    score: float = 0.0


# ============================================================
# 適応的ファジィ決定木マッチャー（プロダクションコード完全準拠）
# ============================================================

class ProductionFuzzyMatcher:
    """
    プロダクションコード（fuzzy_multipath_matcher.py）完全準拠版
    
    ★★★ 実際のシステムと同一のパラメータ ★★★
    """
    
    # ============================================================
    # パラメータ（プロダクションコード準拠）
    # ============================================================
    
    # 1. 類似度計算パラメータ
    SIMILARITY_SIGMA = 0.3  # ★ プロダクションコード準拠（技術資料は0.2）
    
    # 2. 枝刈り閾値
    PRUNING_THRESHOLD = 0.01
    
    # 3. 優先度閾値
    HIGH_PRIORITY_THRESHOLD = 8.0
    MID_PRIORITY_THRESHOLD = 5.0
    
    # 4. 分野マッチングパラメータ（★プロダクションコード準拠★）
    FIELD_EXACT_BONUS = 0.15        # 完全一致ボーナス
    FIELD_MISMATCH_PENALTY = 0.15   # 不一致ペナルティ
    CATEGORY_DECAY = 0.7            # カテゴリ一致減衰
    NO_MATCH_PENALTY = 0.3          # 不一致ベーススコア
    
    # ============================================================
    
    CRITERIA = EVALUATION_CRITERIA
    
    def __init__(self, labs: List[Dict[str, Any]]):
        self.labs = self._normalize_labs(labs)
        self._build_field_category_map()
    
    def _normalize_labs(self, labs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """研究室データを正規化（1-10 → 0-1）"""
        normalized = []
        for lab in labs:
            n_lab = {
                "id": lab["id"],
                "name": lab["name"],
                "field_id": lab["field_id"],
                "professor": lab.get("professor", ""),
                "research_area": lab.get("research_area", ""),
            }
            features = lab.get("features", {})
            for criterion in self.CRITERIA:
                val = features.get(criterion, 5)
                n_lab[criterion] = (val - 1) / 9.0 if val > 1 else 0.0
            normalized.append(n_lab)
        return normalized
    
    def _build_field_category_map(self):
        self.field_to_category = {}
        for cat, fields in FIELD_CATEGORIES.items():
            for f in fields:
                self.field_to_category[f] = cat
    
    def _normalize_value(self, value: float) -> float:
        """1-10を0-1に正規化"""
        if value <= 1:
            return 0.0
        if value > 1:
            return (value - 1) / 9.0
        return value
    
    # ========== Step 1: 優先度ソート ==========
    def _get_sorted_priorities(self, student: Dict[str, Any]) -> List[Dict[str, Any]]:
        priorities = []
        for criterion in self.CRITERIA:
            priority_key = f"{criterion}_priority"
            priority = student.get(priority_key, 5.0)
            priorities.append({"criterion": criterion, "priority": priority})
        priorities.sort(key=lambda x: x["priority"], reverse=True)
        return priorities
    
    # ========== Step 2: 適応的決定木構築 ==========
    def _build_fuzzy_tree(self, priorities: List[Dict[str, Any]]) -> List[Dict]:
        """
        優先度に応じて分岐数を決定:
        - 優先度 ≥ 8: 3分岐（低・中・高）
        - 優先度 5-7: 2分岐（低・高）
        - 優先度 < 5: リーフノード
        """
        tree_layers = []
        
        for item in priorities:
            priority = item["priority"]
            criterion = item["criterion"]
            
            if priority >= self.HIGH_PRIORITY_THRESHOLD:
                tree_layers.append({
                    "criterion": criterion,
                    "priority": priority,
                    "branches": 3,
                    "labels": ["low", "medium", "high"]
                })
            elif priority >= self.MID_PRIORITY_THRESHOLD:
                tree_layers.append({
                    "criterion": criterion,
                    "priority": priority,
                    "branches": 2,
                    "labels": ["low", "high"]
                })
            # 優先度 < 5 はリーフ（決定木には含めない）
        
        return tree_layers
    
    # ========== Step 3: 複数パスの導出 ==========
    def _explore_fuzzy_paths(
        self,
        tree_layers: List[Dict],
        student: Dict[str, Any],
        lab: Dict[str, Any]
    ) -> List[FuzzyPath]:
        if not tree_layers:
            return [FuzzyPath(path_id=0, layers=[], total_membership=1.0, score=0.0)]
        
        all_paths = []
        self._generate_paths_recursive(tree_layers, lab, 0, [], 1.0, all_paths)
        
        # 枝刈り
        pruned_paths = [p for p in all_paths if p.total_membership >= self.PRUNING_THRESHOLD]
        
        if not pruned_paths:
            return [FuzzyPath(path_id=0, layers=[], total_membership=1.0, score=0.0)]
        
        return pruned_paths
    
    def _generate_paths_recursive(
        self,
        tree_layers: List[Dict],
        lab: Dict[str, Any],
        layer_idx: int,
        current_path: List[Tuple[str, str, float]],
        cumulative_membership: float,
        all_paths: List[FuzzyPath]
    ):
        if layer_idx >= len(tree_layers):
            all_paths.append(FuzzyPath(
                path_id=len(all_paths),
                layers=current_path.copy(),
                total_membership=cumulative_membership,
                score=0.0
            ))
            return
        
        layer = tree_layers[layer_idx]
        criterion = layer["criterion"]
        labels = layer["labels"]
        branches = layer["branches"]
        
        lab_value = lab.get(criterion, 0.5)
        
        if branches == 3:
            memberships = MembershipFunctions.fuzzify_3level(lab_value)
        else:
            memberships = MembershipFunctions.fuzzify_2level(lab_value)
        
        for label in labels:
            membership = memberships.get(label, 0.0)
            new_membership = cumulative_membership * membership
            
            if new_membership < self.PRUNING_THRESHOLD:
                continue
            
            new_path = current_path + [(criterion, label, membership)]
            self._generate_paths_recursive(
                tree_layers, lab, layer_idx + 1,
                new_path, new_membership, all_paths
            )
    
    def _normalize_path_memberships(self, paths: List[FuzzyPath]) -> List[FuzzyPath]:
        """パスの所属度を正規化"""
        if not paths:
            return paths
        
        total_membership = sum(path.total_membership for path in paths)
        if total_membership == 0:
            return paths
        
        for path in paths:
            path.total_membership = path.total_membership / total_membership
        
        return paths
    
    # ========== Step 4: 複数パスの統合 ==========
    def _integrate_fuzzy_paths(
        self,
        fuzzy_paths: List[FuzzyPath],
        student: Dict[str, Any],
        lab: Dict[str, Any]
    ) -> Tuple[float, Dict[str, float]]:
        if not fuzzy_paths:
            return 0.5, {}
        
        total_score = 0.0
        all_criteria_scores = {}
        
        for path in fuzzy_paths:
            path_score, criteria_scores = self._calculate_path_score(path, student, lab)
            path.score = path_score
            
            total_score += path.score * path.total_membership
            
            for criterion, score in criteria_scores.items():
                if criterion not in all_criteria_scores:
                    all_criteria_scores[criterion] = []
                all_criteria_scores[criterion].append((score, path.total_membership))
        
        averaged_criteria_scores = {}
        for criterion, score_weight_pairs in all_criteria_scores.items():
            weighted_sum = sum(s * w for s, w in score_weight_pairs)
            averaged_criteria_scores[criterion] = weighted_sum
        
        return total_score, averaged_criteria_scores
    
    def _calculate_path_score(
        self,
        path: FuzzyPath,
        student: Dict[str, Any],
        lab: Dict[str, Any]
    ) -> Tuple[float, Dict[str, float]]:
        """
        パスごとのスコア計算（技術資料 3.5.1節）
        
        ★★★ プロダクションコード準拠: 優先度の非線形変換を適用 ★★★
        """
        criteria_scores = {}
        weighted_sum = 0.0
        total_weight = 0.0
        
        for criterion in self.CRITERIA:
            student_val = self._normalize_value(student.get(criterion, 5.0))
            lab_val = lab.get(criterion, 0.5)
            
            # ガウス類似度計算
            similarity = self._calculate_gaussian_similarity(student_val, lab_val)
            
            # 優先度取得
            priority_key = f"{criterion}_priority"
            priority = student.get(priority_key, 5.0)
            
            # ★★★ 優先度から重みを計算（非線形変換）★★★
            weight = self._calculate_priority_weight(priority)
            
            criteria_scores[criterion] = similarity
            weighted_sum += similarity * weight
            total_weight += weight
        
        path_score = weighted_sum / total_weight if total_weight > 0 else 0.5
        return path_score, criteria_scores
    
    def _calculate_priority_weight(self, priority: float) -> float:
        """
        優先度から重みを計算（非線形変換）
        
        ★★★ プロダクションコード準拠 ★★★
        
        priority 10 → weight 1.00
        priority  8 → weight 0.72
        priority  5 → weight 0.32
        priority  3 → weight 0.11
        priority  1 → weight 0.01
        """
        normalized = priority / 10.0
        weight = normalized ** 1.5  # 非線形変換
        return weight
    
    def _calculate_gaussian_similarity(self, student_val: float, lab_val: float) -> float:
        """
        ガウス類似度計算（技術資料 3.5.2節）
        
        Similarity = exp(-(d²)/(2σ²))
        
        ★ σ = 0.3（プロダクションコード準拠）
        """
        d = abs(student_val - lab_val)
        sigma = self.SIMILARITY_SIGMA
        similarity = math.exp(-(d ** 2) / (2 * sigma ** 2))
        return similarity
    
    # ========== Step 5: 分野マッチング ==========
    def _calculate_field_match(
        self,
        field_interests: Dict[str, float],
        lab_field: str
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Step 5: 分野マッチング（技術資料 3.6節 + プロダクション拡張）
        
        完全一致: I/10
        カテゴリ一致: I/10 × 0.7
        不一致: 0.3
        """
        if not field_interests or not lab_field:
            return 0.5, {"match_type": "unknown", "lab_field": lab_field}
        
        # 完全一致チェック
        if lab_field in field_interests:
            interest_level = field_interests[lab_field]
            score = interest_level / 10.0
            return score, {
                "match_type": "exact",
                "lab_field": lab_field,
                "interest_level": interest_level
            }
        
        # カテゴリ一致チェック
        lab_category = self.field_to_category.get(lab_field, "")
        best_category_score = 0.0
        best_category_field = None
        
        for interest_field, interest_level in field_interests.items():
            interest_category = self.field_to_category.get(interest_field, "")
            if interest_category == lab_category and lab_category:
                category_score = (interest_level / 10.0) * self.CATEGORY_DECAY
                if category_score > best_category_score:
                    best_category_score = category_score
                    best_category_field = interest_field
        
        if best_category_score > 0:
            return best_category_score, {
                "match_type": "category",
                "lab_field": lab_field,
                "matched_interest": best_category_field
            }
        
        # 不一致
        return self.NO_MATCH_PENALTY, {
            "match_type": "no_match",
            "lab_field": lab_field
        }
    
    # ========== Step 6: 最終スコア統合 ==========
    def evaluate(self, student: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        全研究室を評価（Step 1-6 を実行）
        """
        results = []
        
        # research_field_match: 分野重視度
        rfm = student.get("research_field_match", 5.0)
        alpha = self._normalize_value(rfm)
        beta = 1 - alpha
        
        # Step 1: 優先度ソート
        sorted_priorities = self._get_sorted_priorities(student)
        
        # Step 2: 決定木構築
        tree_layers = self._build_fuzzy_tree(sorted_priorities)
        
        for lab in self.labs:
            # Step 3: 複数パスの導出
            fuzzy_paths = self._explore_fuzzy_paths(tree_layers, student, lab)
            
            # 正規化
            fuzzy_paths = self._normalize_path_memberships(fuzzy_paths)
            
            # Step 4: 複数パスの統合
            basic_score, criteria_scores = self._integrate_fuzzy_paths(
                fuzzy_paths, student, lab
            )
            
            # Step 5: 分野マッチング
            field_interests = student.get("field_interests", {})
            field_score, field_detail = self._calculate_field_match(
                field_interests, lab.get("field_id", "")
            )
            
            # Step 6: 最終スコア統合
            # S_final = β × S_basic + α × S_field
            total = beta * basic_score + alpha * field_score
            total = max(0.0, min(1.0, total))
            
            results.append({
                "lab_id": lab["id"],
                "lab_name": lab["name"],
                "field_id": lab["field_id"],
                "professor": lab.get("professor", ""),
                "total_compatibility": total,
                "basic_score": basic_score,
                "field_score": field_score,
                "alpha": alpha,
                "beta": beta,
                "num_paths": len(fuzzy_paths),
                "tree_layers": len(tree_layers),
                "field_match_type": field_detail.get("match_type", "unknown"),
            })
        
        results.sort(key=lambda x: x["total_compatibility"], reverse=True)
        
        for i, r in enumerate(results):
            r["rank"] = i + 1
        
        return results


# ============================================================
# プロファイル生成器
# ============================================================

class StudentProfileGenerator:
    def __init__(self, seed: int = None):
        if seed is not None:
            np.random.seed(seed)
    
    def generate_uniform(self) -> Dict:
        profile = {}
        for criterion in EVALUATION_CRITERIA:
            profile[criterion] = np.random.uniform(1, 10)
            profile[f"{criterion}_priority"] = np.random.uniform(1, 10)
        profile["research_field_match"] = np.random.uniform(1, 10)
        
        num_fields = np.random.randint(1, 4)
        selected = np.random.choice(RESEARCH_FIELDS, num_fields, replace=False)
        profile["field_interests"] = {}
        interests = sorted(np.random.uniform(5, 10, num_fields), reverse=True)
        for f, i in zip(selected, interests):
            profile["field_interests"][f] = i
        
        return profile
    
    def generate_stratified(self, student_type: str, field_category: str = None) -> Dict:
        profile = {}
        
        if student_type == "theory":
            profile["theory_practice"] = np.random.uniform(1, 4)
            profile["research_intensity"] = np.random.uniform(6, 10)
        elif student_type == "practice":
            profile["theory_practice"] = np.random.uniform(7, 10)
            profile["research_intensity"] = np.random.uniform(4, 8)
        else:
            profile["theory_practice"] = np.random.uniform(4, 7)
            profile["research_intensity"] = np.random.uniform(4, 7)
        
        for criterion in EVALUATION_CRITERIA:
            if criterion not in profile:
                profile[criterion] = np.random.uniform(1, 10)
            profile[f"{criterion}_priority"] = np.random.uniform(1, 10)
        
        profile["research_field_match"] = np.random.uniform(1, 10)
        
        if field_category and field_category in FIELD_CATEGORIES:
            available = FIELD_CATEGORIES[field_category]
        else:
            available = RESEARCH_FIELDS
        
        num_fields = np.random.randint(1, min(4, len(available) + 1))
        selected = np.random.choice(available, num_fields, replace=False)
        profile["field_interests"] = {}
        interests = sorted(np.random.uniform(5, 10, num_fields), reverse=True)
        for f, i in zip(selected, interests):
            profile["field_interests"][f] = i
        
        return profile
    
    def generate_boundary(self) -> Dict:
        profile = {}
        boundary_values = [3.0, 5.0, 7.0, 9.0]
        noise = 0.3
        
        for criterion in EVALUATION_CRITERIA:
            base = np.random.choice(boundary_values)
            profile[criterion] = np.clip(base + np.random.uniform(-noise, noise), 1, 10)
            p_base = np.random.choice(boundary_values)
            profile[f"{criterion}_priority"] = np.clip(p_base + np.random.uniform(-noise, noise), 1, 10)
        
        rfm_base = np.random.choice(boundary_values)
        profile["research_field_match"] = np.clip(rfm_base + np.random.uniform(-noise, noise), 1, 10)
        
        num_fields = np.random.randint(1, 4)
        selected = np.random.choice(RESEARCH_FIELDS, num_fields, replace=False)
        profile["field_interests"] = {}
        interests = sorted(np.random.uniform(5, 10, num_fields), reverse=True)
        for f, i in zip(selected, interests):
            profile["field_interests"][f] = i
        
        return profile


# ============================================================
# 公平性分析
# ============================================================

@dataclass
class FairnessMetrics:
    total_samples: int = 0
    lab_first_place_counts: Dict[str, int] = field(default_factory=dict)
    lab_top3_counts: Dict[str, int] = field(default_factory=dict)
    lab_top5_counts: Dict[str, int] = field(default_factory=dict)
    lab_score_sums: Dict[str, float] = field(default_factory=dict)
    lab_score_squared_sums: Dict[str, float] = field(default_factory=dict)
    field_first_place_counts: Dict[str, int] = field(default_factory=dict)
    category_first_place_counts: Dict[str, int] = field(default_factory=dict)
    student_type_avg_options: Dict[str, List[int]] = field(default_factory=dict)
    boundary_rank_changes: List[float] = field(default_factory=list)
    avg_tree_layers: List[int] = field(default_factory=list)
    avg_num_paths: List[int] = field(default_factory=list)


class FairnessAnalyzer:
    def __init__(self, labs: List[Dict]):
        self.labs = labs
        self.matcher = ProductionFuzzyMatcher(labs)
        self.generator = StudentProfileGenerator(seed=42)
        self.metrics = FairnessMetrics()
        
        for lab in labs:
            lid = lab["id"]
            self.metrics.lab_first_place_counts[lid] = 0
            self.metrics.lab_top3_counts[lid] = 0
            self.metrics.lab_top5_counts[lid] = 0
            self.metrics.lab_score_sums[lid] = 0.0
            self.metrics.lab_score_squared_sums[lid] = 0.0
        
        for cat in FIELD_CATEGORIES:
            self.metrics.category_first_place_counts[cat] = 0
    
    def run_phase1_uniform(self, n: int = 20000, interval: int = 5000):
        print(f"\n{'='*60}")
        print(f"Phase 1: 一様サンプリング (N={n:,})")
        print(f"【プロダクションコード準拠マッチャー使用】")
        print(f"{'='*60}")
        
        start = time.time()
        for i in range(n):
            profile = self.generator.generate_uniform()
            results = self.matcher.evaluate(profile)
            self._update_metrics(results)
            self.metrics.total_samples += 1
            
            if (i + 1) % interval == 0:
                elapsed = time.time() - start
                print(f"  進捗: {i+1:,}/{n:,} ({(i+1)/elapsed:.0f} samples/sec)")
        
        print(f"  完了: {time.time()-start:.1f}秒")
    
    def run_phase2_stratified(self, n_per_stratum: int = 1000):
        print(f"\n{'='*60}")
        print(f"Phase 2: 層化サンプリング (各層 N={n_per_stratum:,})")
        print(f"{'='*60}")
        
        student_types = ["theory", "practice", "balanced"]
        field_cats = list(FIELD_CATEGORIES.keys())
        
        for st in student_types:
            self.metrics.student_type_avg_options[st] = []
        
        total = len(student_types) * len(field_cats)
        current = 0
        
        for st in student_types:
            for fc in field_cats:
                current += 1
                n_samples = n_per_stratum // len(field_cats)
                print(f"  [{current}/{total}] {st} × {fc} (N={n_samples})")
                
                for _ in range(n_samples):
                    profile = self.generator.generate_stratified(st, fc)
                    results = self.matcher.evaluate(profile)
                    self._update_metrics(results)
                    self.metrics.total_samples += 1
                    
                    compatible = sum(1 for r in results if r["total_compatibility"] >= 0.6)
                    self.metrics.student_type_avg_options[st].append(compatible)
    
    def run_phase3_boundary(self, n: int = 10000, interval: int = 2000):
        print(f"\n{'='*60}")
        print(f"Phase 3: 境界値テスト (N={n:,})")
        print(f"{'='*60}")
        
        start = time.time()
        for i in range(n):
            profile1 = self.generator.generate_boundary()
            results1 = self.matcher.evaluate(profile1)
            
            profile2 = profile1.copy()
            profile2["field_interests"] = profile1["field_interests"].copy()
            for criterion in EVALUATION_CRITERIA:
                profile2[criterion] = np.clip(
                    profile2[criterion] + np.random.uniform(-0.1, 0.1), 1, 10
                )
            results2 = self.matcher.evaluate(profile2)
            
            changes = []
            for r1 in results1:
                r2 = next((r for r in results2 if r["lab_id"] == r1["lab_id"]), None)
                if r2:
                    changes.append(abs(r1["rank"] - r2["rank"]))
            
            self.metrics.boundary_rank_changes.append(np.mean(changes) if changes else 0)
            self._update_metrics(results1)
            self.metrics.total_samples += 1
            
            if (i + 1) % interval == 0:
                print(f"  進捗: {i+1:,}/{n:,}")
        
        print(f"  完了: {time.time()-start:.1f}秒")
    
    def _update_metrics(self, results: List[Dict]):
        if not results:
            return
        
        first = results[0]
        lid = first["lab_id"]
        fid = first.get("field_id", "unknown")
        
        self.metrics.lab_first_place_counts[lid] += 1
        
        if fid not in self.metrics.field_first_place_counts:
            self.metrics.field_first_place_counts[fid] = 0
        self.metrics.field_first_place_counts[fid] += 1
        
        for cat, fields in FIELD_CATEGORIES.items():
            if fid in fields:
                self.metrics.category_first_place_counts[cat] += 1
                break
        
        for r in results[:3]:
            self.metrics.lab_top3_counts[r["lab_id"]] += 1
        for r in results[:5]:
            self.metrics.lab_top5_counts[r["lab_id"]] += 1
        
        for r in results:
            lid = r["lab_id"]
            score = r["total_compatibility"]
            self.metrics.lab_score_sums[lid] += score
            self.metrics.lab_score_squared_sums[lid] += score ** 2
        
        if results:
            self.metrics.avg_tree_layers.append(results[0].get("tree_layers", 0))
            self.metrics.avg_num_paths.append(results[0].get("num_paths", 0))
    
    def generate_report(self) -> Dict:
        n = self.metrics.total_samples
        num_labs = len(self.labs)
        
        lab_stats = []
        for lab in self.labs:
            lid = lab["id"]
            first_count = self.metrics.lab_first_place_counts.get(lid, 0)
            top3_count = self.metrics.lab_top3_counts.get(lid, 0)
            top5_count = self.metrics.lab_top5_counts.get(lid, 0)
            score_sum = self.metrics.lab_score_sums.get(lid, 0)
            score_sq_sum = self.metrics.lab_score_squared_sums.get(lid, 0)
            
            first_rate = first_count / n if n > 0 else 0
            avg_score = score_sum / n if n > 0 else 0
            variance = (score_sq_sum / n - avg_score**2) if n > 0 else 0
            
            lab_stats.append({
                "lab_id": lid,
                "lab_name": lab["name"],
                "professor": lab.get("professor", ""),
                "field_id": lab["field_id"],
                "research_area": lab.get("research_area", ""),
                "first_place_count": first_count,
                "first_place_rate": first_rate,
                "top3_rate": top3_count / n if n > 0 else 0,
                "top5_rate": top5_count / n if n > 0 else 0,
                "avg_score": avg_score,
                "std_score": math.sqrt(max(0, variance)),
            })
        
        lab_stats.sort(key=lambda x: x["first_place_rate"], reverse=True)
        
        first_rates = [s["first_place_rate"] for s in lab_stats]
        expected = 1.0 / num_labs if num_labs > 0 else 0
        rate_std = np.std(first_rates)
        gini = self._calculate_gini(first_rates)
        
        st_summary = {}
        for st, opts in self.metrics.student_type_avg_options.items():
            if opts:
                st_summary[st] = {
                    "avg_compatible_labs": np.mean(opts),
                    "std_compatible_labs": np.std(opts),
                    "min": int(min(opts)),
                    "max": int(max(opts)),
                    "median": float(np.median(opts)),
                }
        
        boundary = {}
        if self.metrics.boundary_rank_changes:
            changes = self.metrics.boundary_rank_changes
            boundary = {
                "avg_rank_change": float(np.mean(changes)),
                "std_rank_change": float(np.std(changes)),
                "max_rank_change": float(max(changes)),
                "stable_ratio": sum(1 for c in changes if c < 2) / len(changes),
                "very_stable_ratio": sum(1 for c in changes if c < 1) / len(changes),
            }
        
        tree_stats = {}
        if self.metrics.avg_tree_layers:
            tree_stats = {
                "avg_tree_layers": float(np.mean(self.metrics.avg_tree_layers)),
                "avg_num_paths": float(np.mean(self.metrics.avg_num_paths)),
                "max_tree_layers": int(max(self.metrics.avg_tree_layers)),
                "max_num_paths": int(max(self.metrics.avg_num_paths)),
            }
        
        return {
            "experiment_info": {
                "total_samples": n,
                "num_labs": num_labs,
                "num_fields": len(RESEARCH_FIELDS),
                "num_criteria": len(EVALUATION_CRITERIA),
                "algorithm": "適応的ファジィ決定木（プロダクションコード完全準拠）",
                "timestamp": datetime.now().isoformat(),
            },
            "algorithm_parameters": {
                "similarity_sigma": ProductionFuzzyMatcher.SIMILARITY_SIGMA,
                "high_priority_threshold": ProductionFuzzyMatcher.HIGH_PRIORITY_THRESHOLD,
                "mid_priority_threshold": ProductionFuzzyMatcher.MID_PRIORITY_THRESHOLD,
                "pruning_threshold": ProductionFuzzyMatcher.PRUNING_THRESHOLD,
                "category_decay": ProductionFuzzyMatcher.CATEGORY_DECAY,
                "no_match_penalty": ProductionFuzzyMatcher.NO_MATCH_PENALTY,
                "field_exact_bonus": ProductionFuzzyMatcher.FIELD_EXACT_BONUS,
                "field_mismatch_penalty": ProductionFuzzyMatcher.FIELD_MISMATCH_PENALTY,
                "priority_weight_exponent": 1.5,
            },
            "tree_statistics": tree_stats,
            "fairness_summary": {
                "expected_first_rate": expected,
                "actual_rate_std": rate_std,
                "coefficient_of_variation": rate_std / expected if expected > 0 else 0,
                "gini_coefficient": gini,
                "interpretation": self._interpret_gini(gini),
            },
            "lab_statistics": lab_stats,
            "field_distribution": dict(self.metrics.field_first_place_counts),
            "category_distribution": dict(self.metrics.category_first_place_counts),
            "student_type_analysis": st_summary,
            "boundary_stability": boundary,
        }
    
    def _calculate_gini(self, values: List[float]) -> float:
        if not values or sum(values) == 0:
            return 0.0
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        cumsum = np.cumsum(sorted_vals)
        return (2 * sum((i + 1) * v for i, v in enumerate(sorted_vals)) - (n + 1) * cumsum[-1]) / (n * cumsum[-1])
    
    def _interpret_gini(self, gini: float) -> str:
        if gini < 0.2:
            return "非常に公平（推薦機会が均等に分布）"
        elif gini < 0.3:
            return "公平（軽度の偏りあり）"
        elif gini < 0.4:
            return "やや偏りあり（特定研究室が有利）"
        else:
            return "偏りが大きい（要改善）"


# ============================================================
# 結果出力
# ============================================================

def export_to_excel(report: Dict, output_path: str):
    """Excel出力（openpyxlがない場合はCSVにフォールバック）"""
    try:
        import openpyxl
        _export_to_excel_impl(report, output_path)
    except ImportError:
        print("\n⚠️ openpyxlがインストールされていません。CSV形式で出力します。")
        csv_dir = output_path.replace(".xlsx", "_csv")
        export_to_csv(report, csv_dir)


def _export_to_excel_impl(report: Dict, output_path: str):
    """Excel出力の実装"""
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        pd.DataFrame([report["experiment_info"]]).to_excel(writer, sheet_name="実験情報", index=False)
        pd.DataFrame([report["algorithm_parameters"]]).to_excel(writer, sheet_name="アルゴリズム設定", index=False)
        
        if report["tree_statistics"]:
            pd.DataFrame([report["tree_statistics"]]).to_excel(writer, sheet_name="決定木統計", index=False)
        
        pd.DataFrame([report["fairness_summary"]]).to_excel(writer, sheet_name="公平性サマリー", index=False)
        
        lab_df = pd.DataFrame(report["lab_statistics"])
        lab_df.columns = ["研究室ID", "研究室名", "教授", "分野ID", "研究領域",
                          "1位獲得数", "1位獲得率", "Top3率", "Top5率", "平均スコア", "スコア標準偏差"]
        lab_df.to_excel(writer, sheet_name="研究室別統計", index=False)
        
        field_df = pd.DataFrame([{"分野ID": k, "1位獲得数": v} for k, v in report["field_distribution"].items()])
        field_df = field_df.sort_values("1位獲得数", ascending=False)
        field_df.to_excel(writer, sheet_name="分野別分布", index=False)
        
        cat_df = pd.DataFrame([{"カテゴリ": k, "1位獲得数": v} for k, v in report["category_distribution"].items()])
        cat_df = cat_df.sort_values("1位獲得数", ascending=False)
        cat_df.to_excel(writer, sheet_name="カテゴリ別分布", index=False)
        
        if report["student_type_analysis"]:
            type_names = {"theory": "理論志向", "practice": "実践志向", "balanced": "バランス型"}
            st_data = [{"学生タイプ": type_names.get(st, st), "平均適合研究室数": s["avg_compatible_labs"],
                        "標準偏差": s["std_compatible_labs"], "最小": s["min"], "最大": s["max"], "中央値": s["median"]}
                       for st, s in report["student_type_analysis"].items()]
            pd.DataFrame(st_data).to_excel(writer, sheet_name="学生タイプ分析", index=False)
        
        if report["boundary_stability"]:
            bs = report["boundary_stability"]
            pd.DataFrame([{"平均順位変動": bs["avg_rank_change"], "順位変動標準偏差": bs["std_rank_change"],
                           "最大順位変動": bs["max_rank_change"], "安定率(変動<2)": bs["stable_ratio"],
                           "高安定率(変動<1)": bs["very_stable_ratio"]}]).to_excel(writer, sheet_name="境界値安定性", index=False)
    
    print(f"\n✅ Excel出力完了: {output_path}")


def export_to_csv(report: Dict, output_dir: str):
    """CSV形式で出力（openpyxlがない環境用）"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 実験情報
    pd.DataFrame([report["experiment_info"]]).to_csv(
        f"{output_dir}/01_experiment_info.csv", index=False, encoding='utf-8-sig'
    )
    
    # アルゴリズム設定
    pd.DataFrame([report["algorithm_parameters"]]).to_csv(
        f"{output_dir}/02_algorithm_parameters.csv", index=False, encoding='utf-8-sig'
    )
    
    # 決定木統計
    if report["tree_statistics"]:
        pd.DataFrame([report["tree_statistics"]]).to_csv(
            f"{output_dir}/03_tree_statistics.csv", index=False, encoding='utf-8-sig'
        )
    
    # 公平性サマリー
    pd.DataFrame([report["fairness_summary"]]).to_csv(
        f"{output_dir}/04_fairness_summary.csv", index=False, encoding='utf-8-sig'
    )
    
    # 研究室別統計
    lab_df = pd.DataFrame(report["lab_statistics"])
    lab_df.columns = ["研究室ID", "研究室名", "教授", "分野ID", "研究領域",
                      "1位獲得数", "1位獲得率", "Top3率", "Top5率", "平均スコア", "スコア標準偏差"]
    lab_df.to_csv(f"{output_dir}/05_lab_statistics.csv", index=False, encoding='utf-8-sig')
    
    # 分野別分布
    field_df = pd.DataFrame([{"分野ID": k, "1位獲得数": v} for k, v in report["field_distribution"].items()])
    field_df = field_df.sort_values("1位獲得数", ascending=False)
    field_df.to_csv(f"{output_dir}/06_field_distribution.csv", index=False, encoding='utf-8-sig')
    
    # カテゴリ別分布
    cat_df = pd.DataFrame([{"カテゴリ": k, "1位獲得数": v} for k, v in report["category_distribution"].items()])
    cat_df = cat_df.sort_values("1位獲得数", ascending=False)
    cat_df.to_csv(f"{output_dir}/07_category_distribution.csv", index=False, encoding='utf-8-sig')
    
    # 学生タイプ分析
    if report["student_type_analysis"]:
        type_names = {"theory": "理論志向", "practice": "実践志向", "balanced": "バランス型"}
        st_data = [{"学生タイプ": type_names.get(st, st), "平均適合研究室数": s["avg_compatible_labs"],
                    "標準偏差": s["std_compatible_labs"], "最小": s["min"], "最大": s["max"], "中央値": s["median"]}
                   for st, s in report["student_type_analysis"].items()]
        pd.DataFrame(st_data).to_csv(f"{output_dir}/08_student_type_analysis.csv", index=False, encoding='utf-8-sig')
    
    # 境界値安定性
    if report["boundary_stability"]:
        bs = report["boundary_stability"]
        pd.DataFrame([{"平均順位変動": bs["avg_rank_change"], "順位変動標準偏差": bs["std_rank_change"],
                       "最大順位変動": bs["max_rank_change"], "安定率(変動<2)": bs["stable_ratio"],
                       "高安定率(変動<1)": bs["very_stable_ratio"]}]).to_csv(
            f"{output_dir}/09_boundary_stability.csv", index=False, encoding='utf-8-sig'
        )
    
    print(f"\n✅ CSV出力完了: {output_dir}/")


def export_to_json(report: Dict, output_path: str):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"✅ JSON出力完了: {output_path}")


# ============================================================
# メイン実行
# ============================================================

def main():
    print("\n" + "="*70)
    print("モンテカルロ法による公平性検証実験")
    print("【プロダクションコード（fuzzy_multipath_matcher.py）完全準拠版】")
    print("="*70)
    
    print(f"\n■ システム構成")
    print(f"  研究室数: {len(LABS_DATABASE)}")
    print(f"  評価項目数: {len(EVALUATION_CRITERIA)}")
    print(f"  分野数: {len(RESEARCH_FIELDS)}")
    
    print(f"\n■ アルゴリズムパラメータ（★プロダクション準拠★）")
    print(f"  ガウス類似度σ: {ProductionFuzzyMatcher.SIMILARITY_SIGMA}")
    print(f"  高優先度閾値: {ProductionFuzzyMatcher.HIGH_PRIORITY_THRESHOLD}")
    print(f"  中優先度閾値: {ProductionFuzzyMatcher.MID_PRIORITY_THRESHOLD}")
    print(f"  枝刈り閾値: {ProductionFuzzyMatcher.PRUNING_THRESHOLD}")
    print(f"  優先度重み変換: priority^1.5（非線形）")
    print(f"  カテゴリ減衰係数: {ProductionFuzzyMatcher.CATEGORY_DECAY}")
    
    analyzer = FairnessAnalyzer(LABS_DATABASE)
    
    analyzer.run_phase1_uniform(n=20000, interval=5000)
    analyzer.run_phase2_stratified(n_per_stratum=1000)
    analyzer.run_phase3_boundary(n=10000, interval=2000)
    
    report = analyzer.generate_report()
    
    print("\n" + "="*70)
    print("【分析結果サマリー】")
    print("="*70)
    
    exp = report["experiment_info"]
    fair = report["fairness_summary"]
    tree = report["tree_statistics"]
    
    print(f"\n■ 実験規模")
    print(f"  総サンプル数: {exp['total_samples']:,}")
    print(f"  アルゴリズム: {exp['algorithm']}")
    
    print(f"\n■ 決定木統計")
    if tree:
        print(f"  平均レイヤー数: {tree['avg_tree_layers']:.1f}")
        print(f"  平均パス数: {tree['avg_num_paths']:.1f}")
    
    print(f"\n■ 公平性指標")
    print(f"  期待1位獲得率: {fair['expected_first_rate']:.4f} ({fair['expected_first_rate']*100:.2f}%)")
    print(f"  実際の標準偏差: {fair['actual_rate_std']:.4f}")
    print(f"  変動係数 (CV): {fair['coefficient_of_variation']:.4f}")
    print(f"  ジニ係数: {fair['gini_coefficient']:.4f}")
    print(f"  解釈: {fair['interpretation']}")
    
    print(f"\n■ 研究室別1位獲得率 TOP10")
    for i, lab in enumerate(report['lab_statistics'][:10]):
        print(f"  {i+1:2d}. {lab['lab_name']:15s} {lab['first_place_rate']*100:5.2f}% ({lab['first_place_count']:,}回) [{lab['field_id']}]")
    
    print(f"\n■ 研究室別1位獲得率 BOTTOM5")
    for i, lab in enumerate(report['lab_statistics'][-5:]):
        rank = len(report['lab_statistics']) - 4 + i
        print(f"  {rank:2d}. {lab['lab_name']:15s} {lab['first_place_rate']*100:5.2f}% ({lab['first_place_count']:,}回) [{lab['field_id']}]")
    
    if report['boundary_stability']:
        bs = report['boundary_stability']
        print(f"\n■ 境界値安定性（ファジィ理論の効果）")
        print(f"  平均順位変動: {bs['avg_rank_change']:.3f}")
        print(f"  最大順位変動: {bs['max_rank_change']:.3f}")
        print(f"  安定率(変動<2): {bs['stable_ratio']*100:.1f}%")
        print(f"  高安定率(変動<1): {bs['very_stable_ratio']*100:.1f}%")
    
    # 出力ディレクトリ（カレントディレクトリ/outputs に出力）
    import os
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    export_to_excel(report, os.path.join(output_dir, "production_fuzzy_fairness_report.xlsx"))
    export_to_json(report, os.path.join(output_dir, "production_fuzzy_fairness_report.json"))
    
    print("\n" + "="*70)
    print("実験完了")
    print("="*70)
    
    return report


if __name__ == "__main__":
    main()