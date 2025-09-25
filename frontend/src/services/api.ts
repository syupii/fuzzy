// frontend/src/services/api.ts - 優先度対応版APIサービス
/**
 * 遺伝的アルゴリズム×ファジィ決定木×優先度対応 研究室選択支援システム
 * APIサービス v5.0.0
 */

import axios from 'axios';

// API基本設定
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const API_TIMEOUT = 30000; // 30秒

// Axiosインスタンス作成
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: API_TIMEOUT,
  headers: {
    'Content-Type': 'application/json',
  },
});

// レスポンスインターセプター
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error);

    if (error.response?.status === 503) {
      throw new Error('システムが初期化中です。しばらくお待ちください。');
    } else if (error.response?.status >= 500) {
      throw new Error('サーバーエラーが発生しました。システム管理者にお問い合わせください。');
    } else if (error.response?.status === 400) {
      throw new Error(error.response?.data?.detail || '入力データに問題があります。');
    } else if (error.code === 'ECONNABORTED') {
      throw new Error('リクエストがタイムアウトしました。ネットワーク接続を確認してください。');
    } else {
      throw new Error(error.response?.data?.detail || 'APIエラーが発生しました。');
    }
  }
);

// 型定義

// 評価基準（優先度対応）
export interface EvaluationPreferencesWithPriority {
  // 評価値（1-10）
  research_intensity: number;
  advisor_style: number;
  team_work: number;
  workload: number;
  theory_practice: number;
  research_field_match: number;
  skill_development: number;
  lab_atmosphere: number;
  flexibility: number;
  publication_opportunity: number;
  interdisciplinary: number;
  communication_style: number;

  // 優先度オブジェクト（新規）
  priorities: {
    research_intensity: number;
    advisor_style: number;
    team_work: number;
    workload: number;
    theory_practice: number;
    research_field_match: number;
    skill_development: number;
    lab_atmosphere: number;
    flexibility: number;
    publication_opportunity: number;
    interdisciplinary: number;
    communication_style: number;
  };

  // 研究分野興味
  field_interests?: { [key: string]: number };
}

// 学生プロファイル
export interface StudentProfile extends EvaluationPreferencesWithPriority {
  profile_name?: string;
  created_at?: string;
  updated_at?: string;
}

// 優先度分析結果
export interface PriorityAnalysis {
  high_priority_match: number;
  medium_priority_match: number;
  low_priority_match: number;
  priority_distribution: {
    high: number;
    medium: number;
    low: number;
  };
  weighted_priority_score: number;
}

// AI統合スコア
export interface AIScores {
  fuzzy: number;
  genetic: number;
}

// 研究室結果
export interface LabResult {
  lab_id: string;
  lab_name: string;
  advisor: string;
  professor_name: string;
  research_area: string;
  category: string;

  // スコア関連
  final_score: number;
  compatibility_score: number;
  overall_compatibility: number;
  priority_adjusted_score: number;
  base_compatibility: number;

  // AI統合評価
  ai_scores: AIScores;

  // 詳細情報
  feature_scores: { [key: string]: number };
  confidence: number;
  recommendation: string;
  recommendation_level: string;
  explanation: string;

  // 優先度分析
  priority_analysis: PriorityAnalysis | null;

  // ランキング情報
  ranking_position?: number;
}

// 優先度統計
export interface PriorityStatistics {
  average_priority: number;
  max_priority: number;
  min_priority: number;
  priority_variance: number;
  high_priority_count: number;
  medium_priority_count: number;
  low_priority_count: number;
  top_priorities: [string, number][];
}

// 評価サマリー
export interface EvaluationSummary {
  total_labs: number;
  avg_score: number;
  max_score: number;
  min_score: number;
  high_compatibility_count: number;
  medium_compatibility_count: number;
  low_compatibility_count: number;
  priority_weighting_applied: boolean;
  total_priority_items: number;
  priority_statistics: PriorityStatistics | null;
}

// 評価レスポンス
export interface EvaluationResponse {
  lab_results: LabResult[];
  summary: EvaluationSummary;
  student_profile: StudentProfile;
  evaluation_results: LabResult[];
  total_labs_evaluated: number;
  evaluation_timestamp: number;
  metadata: {
    processing_time: number;
    evaluation_count: number;
    priority_evaluations: number;
    timestamp: string;
    criteria_used: number;
    priorities_applied: { [key: string]: number };
    ai_engines_used: string[];
    calculation_method: string;
  };
}

// 研究分野
export interface ResearchField {
  id: string;
  name: string;
  category: string;
  description?: string;
  faculty_count?: number;
  faculty?: string[];
}

// 評価基準情報
export interface CriteriaInfo {
  name: string;
  description: string;
  range: string;
  category: 'basic' | 'extended' | 'special';
}

// システム情報
export interface SystemInfo {
  system_state: any;
  sample_labs_count: number;
  criteria_count: number;
  research_fields_count: number;
  weights: { [key: string]: number };
  has_numpy: boolean;
  priority_support: boolean;
  priority_features: {
    enabled: boolean;
    range: string;
    ai_integration: boolean;
    fuzzy_inference: boolean;
    genetic_algorithm: boolean;
    weighted_scoring: boolean;
  };
  ai_engines: {
    fuzzy: string;
    genetic: string;
  };
  version: string;
}

// 研究室情報
export interface Laboratory {
  id: string;
  name: string;
  advisor: string;
  research_area: string;
  category: string;
  description?: string;

  // 評価基準値
  research_intensity: number;
  advisor_style: number;
  team_work: number;
  workload: number;
  theory_practice: number;
  research_field_match: number;
  skill_development: number;
  lab_atmosphere: number;
  flexibility: number;
  publication_opportunity: number;
  interdisciplinary: number;
  communication_style: number;
}

// フィールドカテゴリ
export const FIELD_CATEGORIES = [
  'テクノロジー・システム',
  'クリエイティブ',
  'エンターテイメント',
  '人文・社会・体育'
];

// 研究分野データ
export const RESEARCH_FIELDS: ResearchField[] = [
  // テクノロジー・システム分野（12分野）
  { id: 'ai_ml', name: '人工知能・機械学習', category: 'テクノロジー・システム', faculty_count: 7 },
  { id: 'image_processing', name: '画像・映像処理', category: 'テクノロジー・システム', faculty_count: 6 },
  { id: 'network_security', name: 'ネットワーク・セキュリティ', category: 'テクノロジー・システム', faculty_count: 3 },
  { id: 'database_systems', name: 'データベース・情報システム', category: 'テクノロジー・システム', faculty_count: 3 },
  { id: 'embedded_iot', name: '組込み・IoT', category: 'テクノロジー・システム', faculty_count: 2 },
  { id: 'education_linguistics', name: '教育・言語学', category: 'テクノロジー・システム', faculty_count: 5 },
  { id: 'natural_science_math', name: '自然科学・数理', category: 'テクノロジー・システム', faculty_count: 6 },
  { id: 'medical_healthcare', name: '医療情報・ヘルスケア', category: 'テクノロジー・システム', faculty_count: 2 },
  { id: 'tourism_regional', name: '観光情報・地域システム', category: 'テクノロジー・システム', faculty_count: 2 },
  { id: 'business_decision', name: '経営情報・意思決定支援', category: 'テクノロジー・システム', faculty_count: 3 },
  { id: 'audio_processing', name: '音声・音響情報処理', category: 'テクノロジー・システム', faculty_count: 2 },
  { id: 'system_ethics', name: 'システム運用・情報倫理', category: 'テクノロジー・システム', faculty_count: 3 },

  // クリエイティブ分野（4分野）
  { id: 'web_design', name: 'Webデザイン・UI/UX', category: 'クリエイティブ', faculty_count: 4 },
  { id: 'design_visual', name: 'デザイン・視覚表現', category: 'クリエイティブ', faculty_count: 4 },
  { id: 'video_animation', name: '映像・アニメーション', category: 'クリエイティブ', faculty_count: 2 },
  { id: 'computer_music', name: 'コンピュータ音楽・サウンドアート', category: 'クリエイティブ', faculty_count: 2 },

  // エンターテイメント分野（2分野）
  { id: 'game_esports', name: 'ゲーム開発・eスポーツ', category: 'エンターテイメント', faculty_count: 2 },
  { id: 'vr_ar_media', name: 'VR/AR・メディアアート', category: 'エンターテイメント', faculty_count: 2 },

  // 人文・社会・体育分野（2分野）
  { id: 'philosophy_humanities', name: '哲学・人文・環境行動学', category: '人文・社会・体育', faculty_count: 2 },
  { id: 'sports_science', name: 'スポーツ・体育科学', category: '人文・社会・体育', faculty_count: 2 }
];

// 評価基準情報
export const CRITERIA_INFO: { [key: string]: CriteriaInfo } = {
  research_intensity: {
    name: '研究強度',
    description: '研究にどれだけ集中的に取り組みたいか',
    range: '1（軽い研究）〜 10（集中研究）',
    category: 'basic'
  },
  advisor_style: {
    name: '指導スタイル',
    description: '教授からの指導の受け方の好み',
    range: '1（厳格指導）〜 10（自由指導）',
    category: 'basic'
  },
  team_work: {
    name: 'チームワーク',
    description: '研究での他者との協働の程度',
    range: '1（個人研究）〜 10（チーム研究）',
    category: 'basic'
  },
  workload: {
    name: 'ワークロード',
    description: '研究活動の忙しさに対する許容度',
    range: '1（軽い負荷）〜 10（重い負荷）',
    category: 'basic'
  },
  theory_practice: {
    name: '理論・実践バランス',
    description: '理論研究と実践的研究のバランス',
    range: '1（理論重視）〜 10（実践重視）',
    category: 'basic'
  },
  research_field_match: {
    name: '研究分野適合性',
    description: '自分の興味と研究室の分野の一致度',
    range: '1（広い分野）〜 10（専門特化）',
    category: 'extended'
  },
  skill_development: {
    name: 'スキル開発',
    description: '専門性と汎用性のバランス',
    range: '1（専門特化）〜 10（幅広いスキル）',
    category: 'extended'
  },
  lab_atmosphere: {
    name: '研究室雰囲気',
    description: '研究室の全体的な雰囲気',
    range: '1（静寂集中）〜 10（活発議論）',
    category: 'extended'
  },
  flexibility: {
    name: '柔軟性',
    description: '研究時間の自由度',
    range: '1（固定スケジュール）〜 10（柔軟スケジュール）',
    category: 'extended'
  },
  publication_opportunity: {
    name: '論文発表機会',
    description: '研究成果の論文化機会',
    range: '1（少ない機会）〜 10（豊富な機会）',
    category: 'extended'
  },
  interdisciplinary: {
    name: '学際性',
    description: '他分野との連携の程度',
    range: '1（単一分野）〜 10（学際連携）',
    category: 'special'
  },
  communication_style: {
    name: 'コミュニケーション',
    description: '研究室での交流スタイル',
    range: '1（少人数密接）〜 10（オープン交流）',
    category: 'special'
  }
};

// APIサービスクラス
class ApiService {
  private baseURL: string;

  constructor() {
    this.baseURL = API_BASE_URL;
  }

  // ヘルスチェック
  async healthCheck(): Promise<{ status: string; message: string; version: string }> {
    try {
      const response = await apiClient.get('/');
      return response.data;
    } catch (error) {
      console.error('Health check failed:', error);
      throw error;
    }
  }

  // システム情報取得
  async getSystemInfo(): Promise<SystemInfo> {
    try {
      const response = await apiClient.get('/api/system');
      return response.data;
    } catch (error) {
      console.error('Failed to get system info:', error);
      throw error;
    }
  }

  // 研究分野一覧取得
  async getResearchFields(): Promise<{ research_fields: any; total_count: number; categories: string[] }> {
    try {
      const response = await apiClient.get('/api/fields');
      return response.data;
    } catch (error) {
      console.error('Failed to get research fields:', error);
      throw error;
    }
  }

  // 評価基準情報取得
  async getEvaluationCriteria(): Promise<{
    criteria: { [key: string]: CriteriaInfo };
    total_count: number;
    categories: any;
    priority_support: boolean;
    priority_range: string;
  }> {
    try {
      const response = await apiClient.get('/api/criteria');
      return response.data;
    } catch (error) {
      console.error('Failed to get evaluation criteria:', error);
      throw error;
    }
  }

  // 研究室一覧取得
  async getLaboratories(): Promise<{ labs: Laboratory[]; total_count: number; categories: string[] }> {
    try {
      const response = await apiClient.get('/api/labs');
      return response.data;
    } catch (error) {
      console.error('Failed to get laboratories:', error);
      throw error;
    }
  }

  // 特定研究室の詳細取得
  async getLaboratoryDetail(labId: string): Promise<Laboratory> {
    try {
      const response = await apiClient.get(`/api/labs/${labId}`);
      return response.data;
    } catch (error) {
      console.error(`Failed to get laboratory detail for ${labId}:`, error);
      throw error;
    }
  }

  // 優先度対応研究室適合度評価（メイン機能）
  async evaluateLabs(evaluationData: any): Promise<EvaluationResponse> {
    try {
      console.log('🚀 優先度対応評価リクエスト送信:', evaluationData);

      const response = await apiClient.post('/api/evaluate', evaluationData);

      console.log('📥 評価レスポンス受信:', response.data);

      // レスポンス検証
      if (!response.data.lab_results || !Array.isArray(response.data.lab_results)) {
        throw new Error('評価結果の形式が正しくありません');
      }

      // 優先度情報の検証
      if (evaluationData.student_profile.priorities && response.data.metadata) {
        console.log('✅ 優先度データが正常に処理されました');
        console.log('🎯 優先度統計:', response.data.summary.priority_statistics);
      }

      return response.data;
    } catch (error) {
      console.error('Failed to evaluate labs with priorities:', error);
      throw error;
    }
  }

  // 遺伝的アルゴリズム最適化（将来実装）
  async optimizeLabAssignments(optimizationData: {
    student_profiles: StudentProfile[];
    constraints?: any;
  }): Promise<any> {
    try {
      const response = await apiClient.post('/api/optimize', optimizationData);
      return response.data;
    } catch (error) {
      console.error('Failed to optimize lab assignments:', error);
      throw error;
    }
  }

  // 詳細説明取得
  async getDetailedExplanation(explanationRequest: {
    student_profile: StudentProfile;
    lab_id: string;
  }): Promise<any> {
    try {
      const response = await apiClient.post('/api/explain', explanationRequest);
      return response.data;
    } catch (error) {
      console.error('Failed to get detailed explanation:', error);
      throw error;
    }
  }

  // プロファイル保存（将来実装）
  async saveStudentProfile(profile: StudentProfile): Promise<{ success: boolean; profile_id: string }> {
    try {
      // 現在はローカルストレージに保存
      const profileId = `profile_${Date.now()}`;
      const profileData = {
        ...profile,
        profile_id: profileId,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString()
      };

      localStorage.setItem(`student_profile_${profileId}`, JSON.stringify(profileData));

      return { success: true, profile_id: profileId };
    } catch (error) {
      console.error('Failed to save student profile:', error);
      throw error;
    }
  }

  // プロファイル読み込み（将来実装）
  async loadStudentProfile(profileId: string): Promise<StudentProfile> {
    try {
      const profileData = localStorage.getItem(`student_profile_${profileId}`);

      if (!profileData) {
        throw new Error('プロファイルが見つかりません');
      }

      return JSON.parse(profileData);
    } catch (error) {
      console.error('Failed to load student profile:', error);
      throw error;
    }
  }

  // 統計情報取得
  async getStatistics(): Promise<{
    total_evaluations: number;
    priority_evaluations: number;
    avg_processing_time: number;
    popular_fields: string[];
    common_priorities: { [key: string]: number };
  }> {
    try {
      // 現在は模擬データを返す
      return {
        total_evaluations: 150,
        priority_evaluations: 89,
        avg_processing_time: 0.124,
        popular_fields: ['ai_ml', 'web_design', 'game_esports'],
        common_priorities: {
          'research_field_match': 8.7,
          'publication_opportunity': 8.1,
          'research_intensity': 7.8,
          'flexibility': 7.2,
          'advisor_style': 6.9
        }
      };
    } catch (error) {
      console.error('Failed to get statistics:', error);
      throw error;
    }
  }
}

// フィールドユーティリティ関数
export const fieldUtils = {
  getFieldsByCategory: (category: string): ResearchField[] => {
    return RESEARCH_FIELDS.filter(field => field.category === category);
  },

  getFieldById: (id: string): ResearchField | undefined => {
    return RESEARCH_FIELDS.find(field => field.id === id);
  },

  getAllCategories: (): string[] => {
    return FIELD_CATEGORIES;
  },

  getFieldCount: (): number => {
    return RESEARCH_FIELDS.length;
  }
};

// 優先度ユーティリティ関数
export const priorityUtils = {
  // 優先度レベル判定
  getPriorityLevel: (priority: number): 'high' | 'medium' | 'low' => {
    if (priority >= 8) return 'high';
    if (priority >= 5) return 'medium';
    return 'low';
  },

  // 優先度統計計算
  calculatePriorityStats: (priorities: { [key: string]: number }) => {
    const values = Object.values(priorities);
    return {
      average: values.reduce((sum, val) => sum + val, 0) / values.length,
      max: Math.max(...values),
      min: Math.min(...values),
      high_count: values.filter(v => v >= 8).length,
      medium_count: values.filter(v => v >= 5 && v < 8).length,
      low_count: values.filter(v => v < 5).length
    };
  },

  // デフォルト優先度生成
  createDefaultPriorities: (): { [key: string]: number } => {
    const defaultPriorities: { [key: string]: number } = {};
    Object.keys(CRITERIA_INFO).forEach(criterion => {
      defaultPriorities[criterion] = 5; // デフォルト値
    });
    return defaultPriorities;
  },

  // 優先度プリセット
  getPresetPriorities: (presetType: 'research_focused' | 'balanced' | 'practical_focused') => {
    const presets = {
      research_focused: {
        research_intensity: 9,
        research_field_match: 10,
        publication_opportunity: 9,
        advisor_style: 6,
        team_work: 5,
        workload: 7,
        theory_practice: 4,
        skill_development: 6,
        lab_atmosphere: 5,
        flexibility: 4,
        interdisciplinary: 7,
        communication_style: 5
      },
      balanced: {
        research_intensity: 7,
        research_field_match: 8,
        publication_opportunity: 7,
        advisor_style: 6,
        team_work: 6,
        workload: 6,
        theory_practice: 6,
        skill_development: 7,
        lab_atmosphere: 6,
        flexibility: 6,
        interdisciplinary: 5,
        communication_style: 6
      },
      practical_focused: {
        research_intensity: 6,
        research_field_match: 7,
        publication_opportunity: 5,
        advisor_style: 8,
        team_work: 8,
        workload: 5,
        theory_practice: 9,
        skill_development: 9,
        lab_atmosphere: 7,
        flexibility: 8,
        interdisciplinary: 6,
        communication_style: 8
      }
    };

    return presets[presetType];
  }
};

// APIサービスインスタンス
export const apiService = new ApiService();

// エクスポート
export default apiService;