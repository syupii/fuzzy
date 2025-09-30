"""
訓練データ生成ユーティリティ
遺伝的アルゴリズムの訓練に使用するサンプルデータを生成
12項目完全対応版
"""

import numpy as np
from typing import List, Dict, Any
import random


class TrainingDataGenerator:
    """訓練データジェネレータ（12項目対応）"""
    
    # 12項目の評価基準
    CRITERIA = [
        "research_intensity", "advisor_style", "team_work",
        "workload", "theory_practice", "research_field_match",
        "skill_development", "lab_atmosphere", "flexibility",
        "publication_opportunity", "interdisciplinary",
        "communication_style"
    ]
    
    # 学生タイプの定義
    STUDENT_TYPES = {
        "high_team_oriented": {
            "research_intensity": (0.7, 1.0),
            "team_work": (0.7, 1.0),
            "workload": (0.7, 1.0),
            "publication_opportunity": (0.7, 1.0)
        },
        "high_individual_focused": {
            "research_intensity": (0.7, 1.0),
            "team_work": (0.0, 0.4),
            "workload": (0.6, 1.0),
            "flexibility": (0.3, 0.6)
        },
        "medium_flexible_style": {
            "research_intensity": (0.4, 0.7),
            "flexibility": (0.6, 1.0),
            "lab_atmosphere": (0.6, 0.9),
            "communication_style": (0.6, 0.9)
        },
        "medium_structured_style": {
            "research_intensity": (0.4, 0.7),
            "flexibility": (0.2, 0.5),
            "workload": (0.4, 0.7),
            "theory_practice": (0.3, 0.6)
        },
        "low_active_atmosphere": {
            "research_intensity": (0.0, 0.4),
            "lab_atmosphere": (0.6, 1.0),
            "communication_style": (0.7, 1.0),
            "team_work": (0.5, 0.8)
        },
        "low_quiet_atmosphere": {
            "research_intensity": (0.0, 0.4),
            "lab_atmosphere": (0.0, 0.4),
            "communication_style": (0.0, 0.4),
            "team_work": (0.2, 0.5)
        }
    }
    
    @classmethod
    def generate_profile_by_type(cls, student_type: str) -> Dict[str, float]:
        """学生タイプに基づいてプロファイルを生成（12項目）
        
        Args:
            student_type: 学生タイプ
            
        Returns:
            プロファイル辞書
        """
        if student_type not in cls.STUDENT_TYPES:
            raise ValueError(f"Unknown student type: {student_type}")
        
        type_ranges = cls.STUDENT_TYPES[student_type]
        profile = {}
        
        # タイプ固有の項目
        for criterion, (min_val, max_val) in type_ranges.items():
            profile[criterion] = np.random.uniform(min_val, max_val)
        
        # その他の項目はランダム（やや中央寄り）
        for criterion in cls.CRITERIA:
            if criterion not in profile:
                # 正規分布で中央寄りに生成
                value = np.random.normal(0.5, 0.15)
                profile[criterion] = np.clip(value, 0.0, 1.0)
        
        return profile
    
    @classmethod
    def generate_random_profile(cls) -> Dict[str, float]:
        """ランダムなプロファイルを生成（12項目）
        
        Returns:
            プロファイル辞書
        """
        profile = {}
        
        for criterion in cls.CRITERIA:
            profile[criterion] = np.random.uniform(0.0, 1.0)
        
        return profile
    
    @classmethod
    def generate_balanced_dataset(cls, samples_per_type: int = 10) -> List[Dict[str, Any]]:
        """バランスの取れた訓練データセットを生成
        
        Args:
            samples_per_type: 各タイプあたりのサンプル数
            
        Returns:
            訓練データのリスト
        """
        training_data = []
        
        for student_type in cls.STUDENT_TYPES.keys():
            for _ in range(samples_per_type):
                profile = cls.generate_profile_by_type(student_type)
                
                training_data.append({
                    "profile": profile,
                    "label": student_type,
                    "score": None
                })
        
        random.shuffle(training_data)
        
        return training_data
    
    @classmethod
    def generate_regression_dataset(cls, 
                                   lab_profiles: List[Dict[str, float]],
                                   num_students: int = 100) -> List[Dict[str, Any]]:
        """回帰タスク用の訓練データを生成
        
        Args:
            lab_profiles: 研究室プロファイルのリスト
            num_students: 生成する学生数
            
        Returns:
            訓練データのリスト
        """
        training_data = []
        
        for _ in range(num_students):
            student_profile = cls.generate_random_profile()
            lab_profile = random.choice(lab_profiles)
            true_score = cls._calculate_true_similarity(student_profile, lab_profile)
            
            training_data.append({
                "profile": student_profile,
                "lab_profile": lab_profile,
                "score": true_score,
                "label": None
            })
        
        return training_data
    
    @classmethod
    def _calculate_true_similarity(cls, 
                                   student: Dict[str, float],
                                   lab: Dict[str, float]) -> float:
        """真の類似度を計算（12項目）
        
        Args:
            student: 学生プロファイル
            lab: 研究室プロファイル
            
        Returns:
            類似度スコア（0〜1）
        """
        total = 0.0
        
        # 重要度（固定）- 12項目
        weights = {
            "research_intensity": 0.15,
            "research_field_match": 0.13,
            "advisor_style": 0.10,
            "team_work": 0.09,
            "publication_opportunity": 0.08,
            "workload": 0.07,
            "skill_development": 0.07,
            "lab_atmosphere": 0.07,
            "flexibility": 0.07,
            "theory_practice": 0.06,
            "interdisciplinary": 0.06,
            "communication_style": 0.05
        }
        
        for criterion in cls.CRITERIA:
            student_val = student.get(criterion, 0.5)
            lab_val = lab.get(criterion, 0.5)
            weight = weights.get(criterion, 0.05)
            
            # ガウス型類似度
            diff = abs(student_val - lab_val)
            similarity = np.exp(-0.5 * (diff / 0.2) ** 2)
            
            total += similarity * weight
        
        return total
    
    @classmethod
    def generate_mixed_dataset(cls,
                              lab_profiles: List[Dict[str, float]],
                              classification_samples: int = 60,
                              regression_samples: int = 40) -> List[Dict[str, Any]]:
        """分類と回帰の混合データセットを生成
        
        Args:
            lab_profiles: 研究室プロファイルのリスト
            classification_samples: 分類タスクのサンプル数
            regression_samples: 回帰タスクのサンプル数
            
        Returns:
            混合訓練データのリスト
        """
        # 分類データ
        classification_data = cls.generate_balanced_dataset(
            samples_per_type=classification_samples // 6
        )
        
        # 回帰データ
        regression_data = cls.generate_regression_dataset(
            lab_profiles=lab_profiles,
            num_students=regression_samples
        )
        
        # 統合
        mixed_data = classification_data + regression_data
        random.shuffle(mixed_data)
        
        return mixed_data
    
    @classmethod
    def add_noise_to_profile(cls, 
                            profile: Dict[str, float],
                            noise_level: float = 0.1) -> Dict[str, float]:
        """プロファイルにノイズを追加
        
        Args:
            profile: 元のプロファイル
            noise_level: ノイズレベル（標準偏差）
            
        Returns:
            ノイズ付きプロファイル
        """
        noisy_profile = {}
        
        for criterion, value in profile.items():
            noise = np.random.normal(0, noise_level)
            noisy_value = np.clip(value + noise, 0.0, 1.0)
            noisy_profile[criterion] = noisy_value
        
        return noisy_profile
    
    @classmethod
    def generate_augmented_dataset(cls,
                                  base_dataset: List[Dict[str, Any]],
                                  augmentation_factor: int = 3,
                                  noise_level: float = 0.05) -> List[Dict[str, Any]]:
        """データ拡張
        
        Args:
            base_dataset: 元のデータセット
            augmentation_factor: 拡張倍率
            noise_level: ノイズレベル
            
        Returns:
            拡張されたデータセット
        """
        augmented_data = base_dataset.copy()
        
        for _ in range(augmentation_factor - 1):
            for data in base_dataset:
                noisy_profile = cls.add_noise_to_profile(
                    data["profile"], 
                    noise_level
                )
                
                augmented_sample = data.copy()
                augmented_sample["profile"] = noisy_profile
                augmented_data.append(augmented_sample)
        
        random.shuffle(augmented_data)
        
        return augmented_data
    
    @classmethod
    def generate_with_field_interests(cls,
                                     samples_per_type: int = 10,
                                     fields: List[str] = None) -> List[Dict[str, Any]]:
        """分野興味度を含む訓練データを生成
        
        Args:
            samples_per_type: 各タイプあたりのサンプル数
            fields: 研究分野リスト
            
        Returns:
            訓練データのリスト
        """
        if fields is None:
            fields = [
                "人工知能・機械学習", "画像・映像処理", "ネットワーク・セキュリティ",
                "データベース・情報システム", "組込み・IoT", "教育・言語学",
                "自然科学・数理", "観光情報・地域システム", "経営情報・意思決定支援",
                "音声・音響情報処理", "システム運用・情報倫理",
                "Webデザイン・UI/UX", "デザイン・視覚表現", "映像・アニメーション",
                "コンピュータ音楽・サウンドアート", "ゲーム開発・eスポーツ",
                "VR/AR・メディアアート", "哲学・人文・環境行動学", "スポーツ・体育科学"
            ]
        
        training_data = []
        
        for student_type in cls.STUDENT_TYPES.keys():
            for _ in range(samples_per_type):
                profile = cls.generate_profile_by_type(student_type)
                
                # 分野興味度を生成（正規分布）
                field_interests = {}
                for field in fields:
                    interest = np.random.beta(2, 2)  # Beta分布で0〜1の値を生成
                    field_interests[field] = float(np.clip(interest, 0.0, 1.0))
                
                profile["field_interests"] = field_interests
                
                training_data.append({
                    "profile": profile,
                    "label": student_type,
                    "score": None
                })
        
        random.shuffle(training_data)
        
        return training_data


# 使用例とテスト
if __name__ == "__main__":
    print("=" * 70)
    print("訓練データ生成テスト（12項目対応）")
    print("=" * 70)
    
    generator = TrainingDataGenerator()
    
    # タイプ別プロファイル生成
    print("\n1. 学生タイプ別プロファイル（12項目）")
    for student_type in list(generator.STUDENT_TYPES.keys())[:2]:
        profile = generator.generate_profile_by_type(student_type)
        print(f"\n{student_type}:")
        print(f"  research_intensity: {profile['research_intensity']:.2f}")
        print(f"  team_work: {profile['team_work']:.2f}")
        print(f"  flexibility: {profile['flexibility']:.2f}")
        print(f"  interdisciplinary: {profile['interdisciplinary']:.2f}")
    
    # バランスデータセット生成
    print("\n2. バランス訓練データセット")
    balanced_data = generator.generate_balanced_dataset(samples_per_type=5)
    print(f"総サンプル数: {len(balanced_data)}")
    
    # タイプ分布確認
    type_counts = {}
    for data in balanced_data:
        label = data["label"]
        type_counts[label] = type_counts.get(label, 0) + 1
    
    print("タイプ分布:")
    for label, count in type_counts.items():
        print(f"  {label}: {count}サンプル")
    
    # 分野興味度付きデータ
    print("\n3. 分野興味度付き訓練データ")
    field_data = generator.generate_with_field_interests(samples_per_type=2)
    print(f"総サンプル数: {len(field_data)}")
    
    sample = field_data[0]
    print(f"\nサンプル例:")
    print(f"  タイプ: {sample['label']}")
    print(f"  12項目設定済み: {len(sample['profile']) - 1}")  # field_interests除く
    print(f"  分野興味度設定済み: {len(sample['profile']['field_interests'])}分野")
    
    # トップ3興味分野を表示
    interests = sample['profile']['field_interests']
    top3 = sorted(interests.items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"  トップ3興味分野:")
    for field, interest in top3:
        print(f"    {field}: {interest:.2f}")
    
    print("\n" + "=" * 70)
    print("✅ 12項目+19分野対応訓練データ生成 - テスト完了")