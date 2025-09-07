// src/services/api.ts - 完全修正版
import axios from 'axios';

const API_BASE_URL = process.env.NODE_ENV === 'production'
  ? '/api'
  : 'http://localhost:8000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// 13項目評価基準（バックエンド仕様に合わせる）
export interface EvaluationPreferences {
  // 基本項目（5項目）
  research_intensity: number;    // 研究強度 (1-10)
  advisor_style: number;        // 指導スタイル (1-10)
  team_work: number;           // チームワーク (1-10)
  workload: number;            // ワークロード (1-10)
  theory_practice: number;     // 理論・実践バランス (1-10)

  // 拡張項目（5項目）
  research_field_match: number;    // 研究分野適合性 (1-10)
  skill_development: number;       // スキル開発 (1-10)
  lab_atmosphere: number;          // 研究室雰囲気 (1-10)
  flexibility: number;             // 柔軟性 (1-10)
  publication_opportunity: number; // 論文発表機会 (1-10)

  // 特殊項目（3項目）
  interdisciplinary: number;       // 学際性 (1-10)
  communication_style: number;     // コミュニケーション (1-10)
  innovation_risk: number;         // 革新性・リスク許容度 (1-10)
}

// 研究分野の興味度
export interface ResearchFieldInterests {
  "人工知能・機械学習": number;
  "画像・映像処理": number;
  "コンピュータネットワーク・セキュリティ": number;
  "データベース・情報システム": number;
  "組込み・IoT": number;
  "Webデザイン・UI/UX": number;
  "デザイン・視覚表現": number;
  "映像・アニメーション": number;
  "コンピュータ音楽・サウンドアート": number;
  "ゲーム開発・eスポーツ": number;
  "VR/AR・メディアアート": number;
}

export interface StudentProfile {
  preferences: EvaluationPreferences;
  field_interests: ResearchFieldInterests;
  metadata?: {
    timestamp?: number;
    session_id?: string;
  };
}

export interface LabResult {
  lab_id: string;
  lab_name: string;
  advisor: string;
  description: string;
  overall_score: number;
  rank: number;
  detailed_scores: {
    [criteriaName: string]: number;
  };
  field_compatibility: number;
  strengths: string[];
  considerations: string[];
  research_areas: string[];
  facilities?: string;
  publications?: number;
  funding?: string;
}

export interface EvaluationSummary {
  total_labs_evaluated: number;
  avg_score: number;
  top_score: number;
  criteria_analysis: {
    [criteriaName: string]: {
      weight: number;
      avg_value: number;
      impact_score: number;
    };
  };
  field_analysis: {
    selected_fields_count: number;
    primary_interests: string[];
    field_distribution: {
      [fieldName: string]: number;
    };
  };
  recommendations: string[];
}

export interface EvaluationResponse {
  evaluation_id: string;
  results: LabResult[];
  summary: EvaluationSummary;
  student_profile: StudentProfile;
  processing_time: number;
  algorithm_info: {
    method: string;
    data_source: string;
    fuzzy_available: boolean;
    genetic_available: boolean;
    decision_tree_available: boolean;
  };
}

// 研究分野定義
export const RESEARCH_FIELDS = [
  "人工知能・機械学習",
  "画像・映像処理",
  "コンピュータネットワーク・セキュリティ",
  "データベース・情報システム",
  "組込み・IoT",
  "Webデザイン・UI/UX",
  "デザイン・視覚表現",
  "映像・アニメーション",
  "コンピュータ音楽・サウンドアート",
  "ゲーム開発・eスポーツ",
  "VR/AR・メディアアート"
] as const;

export const FIELD_CATEGORIES = {
  "テクノロジー・システム": [
    "人工知能・機械学習",
    "画像・映像処理",
    "コンピュータネットワーク・セキュリティ",
    "データベース・情報システム",
    "組込み・IoT"
  ],
  "クリエイティブ": [
    "Webデザイン・UI/UX",
    "デザイン・視覚表現",
    "映像・アニメーション",
    "コンピュータ音楽・サウンドアート"
  ],
  "エンターテイメント": [
    "ゲーム開発・eスポーツ",
    "VR/AR・メディアアート"
  ]
};

// 評価基準の詳細情報
export const CRITERIA_INFO = {
  // 基本項目（5項目）
  research_intensity: {
    label: "研究強度",
    description: "研究にどれだけ集中的に取り組みたいか",
    range: "1（軽い研究）〜 10（集中研究）",
    category: "basic"
  },
  advisor_style: {
    label: "指導スタイル",
    description: "教授からの指導の受け方の好み",
    range: "1（厳格指導）〜 10（自由指導）",
    category: "basic"
  },
  team_work: {
    label: "チームワーク",
    description: "研究での他者との協働の程度",
    range: "1（個人研究）〜 10（チーム研究）",
    category: "basic"
  },
  workload: {
    label: "ワークロード",
    description: "研究活動の忙しさに対する許容度",
    range: "1（軽い負荷）〜 10（重い負荷）",
    category: "basic"
  },
  theory_practice: {
    label: "理論・実践バランス",
    description: "理論研究と実践的研究のバランス",
    range: "1（理論重視）〜 10（実践重視）",
    category: "basic"
  },

  // 拡張項目（5項目）
  research_field_match: {
    label: "研究分野適合性",
    description: "自分の興味と研究室の分野の一致度",
    range: "1（広い分野）〜 10（専門特化）",
    category: "extended"
  },
  skill_development: {
    label: "スキル開発",
    description: "専門性と汎用性のバランス",
    range: "1（専門特化）〜 10（幅広いスキル）",
    category: "extended"
  },
  lab_atmosphere: {
    label: "研究室雰囲気",
    description: "研究室の全体的な雰囲気",
    range: "1（静寂集中）〜 10（活発議論）",
    category: "extended"
  },
  flexibility: {
    label: "柔軟性",
    description: "研究時間の自由度",
    range: "1（固定スケジュール）〜 10（柔軟スケジュール）",
    category: "extended"
  },
  publication_opportunity: {
    label: "論文発表機会",
    description: "研究成果の論文化機会",
    range: "1（少ない機会）〜 10（豊富な機会）",
    category: "extended"
  },

  // 特殊項目（3項目）
  interdisciplinary: {
    label: "学際性",
    description: "他分野との連携の程度",
    range: "1（単一分野）〜 10（学際連携）",
    category: "special"
  },
  communication_style: {
    label: "コミュニケーション",
    description: "研究室での交流スタイル",
    range: "1（少人数密接）〜 10（オープン交流）",
    category: "special"
  },
  innovation_risk: {
    label: "革新性・リスク許容度",
    description: "新しい手法への挑戦度",
    range: "1（安全手法）〜 10（革新手法）",
    category: "special"
  }
};

// ユーティリティ関数
export const fieldUtils = {
  getCategoryForField: (field: string): string => {
    for (const [category, fields] of Object.entries(FIELD_CATEGORIES)) {
      if (fields.includes(field)) {
        return category;
      }
    }
    return "その他";
  },

  getFieldsByCategory: (category: string): string[] => {
    return FIELD_CATEGORIES[category as keyof typeof FIELD_CATEGORIES] || [];
  },

  getAllFields: (): string[] => {
    return [...RESEARCH_FIELDS];
  }
};

// システム統計情報の型定義（ローカル）
interface SystemStatsResponse {
  total_evaluations: number;
  avg_score: number;
  popular_criteria: string[];
  lab_rankings: Array<{
    lab_name: string;
    avg_score: number;
    evaluation_count: number;
  }>;
}

// API サービス
export const apiService = {
  async evaluateLabs(studentProfile: StudentProfile): Promise<EvaluationResponse> {
    try {
      const response = await api.post('/evaluate', studentProfile);
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error)) {
        throw new Error(error.response?.data?.detail || 'API通信エラーが発生しました');
      }
      throw new Error('予期しないエラーが発生しました');
    }
  },

  async getHealthStatus(): Promise<any> {
    try {
      const response = await api.get('/health');
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error)) {
        throw new Error(error.response?.data?.detail || 'ヘルスチェックに失敗しました');
      }
      throw new Error('システム状態の確認に失敗しました');
    }
  },

  async getConfig(): Promise<any> {
    try {
      const response = await api.get('/config');
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error)) {
        throw new Error(error.response?.data?.detail || '設定情報の取得に失敗しました');
      }
      throw new Error('設定情報の取得に失敗しました');
    }
  },

  async getSystemStats(): Promise<SystemStatsResponse> {
    try {
      const response = await api.get('/stats');
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error)) {
        throw new Error(error.response?.data?.detail || 'システム統計の取得に失敗しました');
      }
      throw new Error('システム統計の取得に失敗しました');
    }
  }
};

export default apiService;