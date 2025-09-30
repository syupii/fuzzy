# utils/training_data.py - 訓練データ生成ユーティリティ

import numpy as np
from typing import List, Dict, Any
import random

class TrainingDataGenerator:
    """遺伝的アルゴリズム用の訓練データ生成クラス"""
    
    # 13項目評価基準
    CRITERIA = [
        "research_intensity", "advisor_style", "team_work", 
        "workload", "theory_practice",
        "research_field_match", "skill_development", "lab_atmosphere",
        "flexibility", "publication_opportunity",
        "interdisciplinary", "communication_style", "innovation_risk"
    ]
    
    # 学生タイプ定義
    STUDENT_TYPES = {
        "研究志向型": {
            "research_intensity": (7.5, 10.0),
            "advisor_style": (4.0, 7.0),
            "team_work": (5.0, 8.0),
            "workload": (7.0, 10.0),
            "theory_practice": (3.0, 6.0),
            "research_field_match": (8.0, 10.0),
            "skill_development": (6.0, 9.0),
            "lab_atmosphere": (5.0, 8.0),
            "flexibility": (4.0, 7.0),
            "publication_opportunity": (8.0, 10.0),
            "interdisciplinary": (5.0, 8.0),
            "communication_style": (5.0, 8.0),
            "innovation_risk": (6.0, 9.0)
        },
        "実践志向型": {
            "research_intensity": (5.0, 8.0),
            "advisor_style": (6.0, 9.0),
            "team_work": (7.0, 10.0),
            "workload": (5.0, 8.0),
            "theory_practice": (7.0, 10.0),
            "research_field_match": (6.0, 9.0),
            "skill_development": (7.0, 10.0),
            "lab_atmosphere": (7.0, 10.0),
            "flexibility": (7.0, 10.0),
            "publication_opportunity": (4.0, 7.0),
            "interdisciplinary": (6.0, 9.0),
            "communication_style": (7.0, 10.0),
            "innovation_risk": (5.0, 8.0)
        },
        "バランス型": {
            "research_intensity": (5.0, 7.5),
            "advisor_style": (5.0, 8.0),
            "team_work": (5.0, 8.0),
            "workload": (5.0, 7.5),
            "theory_practice": (4.5, 7.0),
            "research_field_match": (6.0, 8.5),
            "skill_development": (6.0, 8.5),
            "lab_atmosphere": (5.5, 8.0),
            "flexibility": (5.5, 8.0),
            "publication_opportunity": (5.5, 8.0),
            "interdisciplinary": (5.5, 8.0),
            "communication_style": (5.5, 8.0),
            "innovation_risk": (5.0, 7.5)
        },
        "自由志向型": {
            "research_intensity": (4.0, 7.0),
            "advisor_style": (7.0, 10.0),
            "team_work": (4.0, 7.0),
            "workload": (3.0, 6.0),
            "theory_practice": (5.0, 8.0),
            "research_field_match": (5.0, 8.0),
            "skill_development": (6.0, 9.0),
            "lab_atmosphere": (6.0, 9.0),
            "flexibility": (8.0, 10.0),
            "publication_opportunity": (4.0, 7.0),
            "interdisciplinary": (7.0, 10.0),
            "communication_style": (7.0, 10.0),
            "innovation_risk": (6.0, 9.0)
        },
        "チーム志向型": {
            "research_intensity": (5.0, 8.0),
            "advisor_style": (5.0, 8.0),
            "team_work": (8.0, 10.0),
            "workload": (6.0, 9.0),
            "theory_practice": (5.0, 8.0),
            "research_field_match": (6.0, 9.0),
            "skill_development": (7.0, 10.0),
            "lab_atmosphere": (8.0, 10.0),
            "flexibility": (5.0, 8.0),
            "publication_opportunity": (6.0, 9.0),
            "interdisciplinary": (7.0, 10.0),
            "communication_style": (8.0, 10.0),
            "innovation_risk": (5.0, 8.0)
        },
        "専門特化型": {
            "research_intensity": (7.0, 10.0),
            "advisor_style": (4.0, 7.0),
            "team_work": (4.0, 7.0),
            "workload": (7.0, 10.0),
            "theory_practice": (3.0, 6.0),
            "research_field_match": (8.5, 10.0),
            "skill_development": (5.0, 7.5),
            "lab_atmosphere": (4.0, 7.0),
            "flexibility": (4.0, 7.0),
            "publication_opportunity": (8.0, 10.0),
            "interdisciplinary": (3.0, 6.0),
            "communication_style": (4.0, 7.0),
            "innovation_risk": (7.0, 10.0)
        }
    }
    
    @classmethod
    def generate_balanced_dataset(cls, samples_per_type: int = 20) -> List[Dict[str, Any]]:
        """
        バランスの取れた訓練データセット生成
        
        Args:
            samples_per_type: 各学生タイプあたりのサンプル数
            
        Returns:
            訓練データのリスト
        """
        
        training_data = []
        
        for student_type, ranges in cls.STUDENT_TYPES.items():
            for _ in range(samples_per_type):
                # プロファイル生成
                profile = {}
                for criterion in cls.CRITERIA:
                    min_val, max_val = ranges[criterion]
                    profile[criterion] = random.uniform(min_val, max_val)
                
                # ラベル生成（クラスタリング）
                label = cls._generate_label(profile, student_type)
                
                training_data.append({
                    "profile": profile,
                    "label": label,
                    "student_type": student_type
                })
        
        # シャッフル
        random.shuffle(training_data)
        
        return training_data
    
    @classmethod
    def _generate_label(cls, profile: Dict[str, float], student_type: str) -> str:
        """プロファイルから適切なクラスタラベルを生成"""
        
        # 研究強度ベース
        research_intensity = profile["research_intensity"]
        team_work = profile["team_work"]
        flexibility = profile["flexibility"]
        
        # 複合的な判定
        if research_intensity > 7.5:
            if team_work > 7.5:
                return "high_research_high_team"
            else:
                return "high_research_low_team"
        elif research_intensity > 5.5:
            if flexibility > 7.0:
                return "medium_research_high_flex"
            else:
                return "medium_research_low_flex"
        else:
            if team_work > 7.0:
                return "low_research_high_team"
            else:
                return "low_research_low_team"
    
    @classmethod
    def generate_custom_dataset(
        cls, 
        profiles: List[Dict[str, float]],
        labels: List[str]
    ) -> List[Dict[str, Any]]:
        """
        カスタム訓練データセット生成
        
        Args:
            profiles: 学生プロファイルのリスト
            labels: 対応するラベルのリスト
            
        Returns:
            訓練データのリスト
        """
        
        if len(profiles) != len(labels):
            raise ValueError("Profiles and labels must have same length")
        
        training_data = []
        for profile, label in zip(profiles, labels):
            training_data.append({
                "profile": profile,
                "label": label,
                "student_type": "custom"
            })
        
        return training_data
    
    @classmethod
    def add_noise(cls, dataset: List[Dict[str, Any]], noise_level: float = 0.1) -> List[Dict[str, Any]]:
        """
        データセットにノイズを追加（データ拡張）
        
        Args:
            dataset: 元のデータセット
            noise_level: ノイズレベル（0-1）
            
        Returns:
            ノイズ追加後のデータセット
        """
        
        noisy_dataset = []
        
        for data in dataset:
            noisy_profile = {}
            for criterion, value in data["profile"].items():
                # ガウシアンノイズ追加
                noise = np.random.normal(0, noise_level)
                noisy_value = value + noise
                # 範囲制限 (0-10)
                noisy_profile[criterion] = max(0.0, min(10.0, noisy_value))
            
            noisy_dataset.append({
                "profile": noisy_profile,
                "label": data["label"],
                "student_type": data.get("student_type", "unknown")
            })
        
        return noisy_dataset
    
    @classmethod
    def generate_test_profiles(cls, count: int = 10) -> List[Dict[str, float]]:
        """
        テスト用プロファイル生成
        
        Args:
            count: 生成するプロファイル数
            
        Returns:
            テストプロファイルのリスト
        """
        
        test_profiles = []
        
        for _ in range(count):
            # ランダムに学生タイプを選択
            student_type = random.choice(list(cls.STUDENT_TYPES.keys()))
            ranges = cls.STUDENT_TYPES[student_type]
            
            profile = {}
            for criterion in cls.CRITERIA:
                min_val, max_val = ranges[criterion]
                profile[criterion] = random.uniform(min_val, max_val)
            
            test_profiles.append(profile)
        
        return test_profiles


# 使用例・テスト
if __name__ == "__main__":
    print("=" * 70)
    print("訓練データ生成テスト")
    print("=" * 70)
    
    # バランスデータセット生成
    print("\n📊 バランスデータセット生成中...")
    dataset = TrainingDataGenerator.generate_balanced_dataset(samples_per_type=5)
    
    print(f"✅ 生成完了: {len(dataset)}サンプル")
    print(f"\n📋 学生タイプ分布:")
    type_counts = {}
    for data in dataset:
        student_type = data["student_type"]
        type_counts[student_type] = type_counts.get(student_type, 0) + 1
    
    for student_type, count in type_counts.items():
        print(f"  {student_type}: {count}サンプル")
    
    print(f"\n📋 ラベル分布:")
    label_counts = {}
    for data in dataset:
        label = data["label"]
        label_counts[label] = label_counts.get(label, 0) + 1
    
    for label, count in label_counts.items():
        print(f"  {label}: {count}サンプル")
    
    # サンプル表示
    print(f"\n📄 サンプルデータ（最初の3件）:")
    for i, data in enumerate(dataset[:3]):
        print(f"\n  サンプル {i+1}:")
        print(f"    学生タイプ: {data['student_type']}")
        print(f"    ラベル: {data['label']}")
        print(f"    プロファイル（抜粋）:")
        for criterion in ['research_intensity', 'team_work', 'flexibility']:
            print(f"      {criterion}: {data['profile'][criterion]:.2f}")
    
    print("\n" + "=" * 70)