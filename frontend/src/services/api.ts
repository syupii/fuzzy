// src/services/api.ts - 完全修正版
import axios from 'axios';

// 基本的な型定義
export interface EvaluationPreferences {
  // 基本項目
  research_intensity: number;
  advisor_style: number;
  team_work: number;
  workload: number;
  theory_practice: number;

  // 拡張項目
  research_field_match: number;
  skill_development: number;
  learning_pace: number;
  difficulty_preference: number;
  communication_style: number;
  meeting_frequency: number;
  lab_atmosphere: number;
  innovation_risk: number;
  methodology_preference: number;
  interdisciplinary: number;
  flexibility: number;
  evening_weekend_work: number;
  publication_opportunity: number;
  financial_support: number;
  lab_hierarchy: number;
  core_time_flexibility: number;
}

export interface Laboratory {
  id: string;
  name: string;
  professor: string;
  research_area: string;
  specialization: string;
  description?: string;
}

export interface CompatibilityScore {
  overall_score: number;
  criterion_scores: {
    [key: string]: {
      similarity: number;
      weight: number;
      score: number;
    };
  };
}

export interface LabResult {
  lab: Laboratory;
  compatibility: CompatibilityScore;
  ranking_position: number;
  recommendations?: string[];
}

export interface FieldAnalysis {
  selected_fields_count: number;
  field_distribution: {
    [key: string]: number;
  };
}

export interface EvaluationSummary {
  total_labs: number;
  avg_score: number;
  best_match_lab?: string;
  avg_compatibility_score?: number;
  field_analysis?: FieldAnalysis;
}

export interface EvaluationResponse {
  results: LabResult[];
  summary: EvaluationSummary;
}

// 研究分野関連の型定義
export interface ResearchField {
  id: string;
  name: string;
  category: string;
  description: string;
  icon: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  keywords: string[];
}

export interface FieldInterest {
  fieldId: string;
  isSelected: boolean;
  interestLevel: number;
}

// 研究分野の興味度マップ
export interface ResearchFieldInterests {
  [fieldName: string]: number;
}

// 学生プロフィール
export interface StudentProfile {
  evaluation_criteria: EvaluationPreferences;
  field_interests: ResearchFieldInterests;
}

// 技術スタック関連の型定義
export interface ProgrammingLanguage {
  id: string;
  name: string;
  category: string;
  description: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  icon: string;
  popularity: number;
}

export interface TechFramework {
  id: string;
  name: string;
  category: string;
  description: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  relatedLanguages: string[];
  icon: string;
}

export interface TechStackPreference {
  languageId: string;
  experienceLevel: number;
  interest: number;
}

// 研究分野データ
export const RESEARCH_FIELDS: ResearchField[] = [
  // 人工知能・機械学習分野
  {
    id: 'ai_ml',
    name: '人工知能・機械学習',
    category: 'テクノロジー・システム',
    description: 'AI、機械学習、データ解析、レコメンドシステム',
    icon: '🤖',
    difficulty: 'intermediate',
    keywords: ['AI', '機械学習', 'データ解析', 'レコメンド', 'マルチエージェント']
  },
  {
    id: 'image_video',
    name: '画像・映像処理',
    category: 'テクノロジー・システム',
    description: '画像処理、コンピュータビジョン、3DCG、VR/AR',
    icon: '📸',
    difficulty: 'intermediate',
    keywords: ['画像処理', 'コンピュータビジョン', '3DCG', 'VR', 'AR', '医用画像']
  },
  {
    id: 'network_security',
    name: 'コンピュータネットワーク・セキュリティ',
    category: 'テクノロジー・システム',
    description: 'ネットワーク技術、情報セキュリティ、通信システム',
    icon: '🔒',
    difficulty: 'advanced',
    keywords: ['ネットワーク', 'セキュリティ', '通信システム', 'ITマネジメント']
  },
  {
    id: 'database_systems',
    name: 'データベース・情報システム',
    category: 'テクノロジー・システム',
    description: 'データベース技術、経営情報システム、意思決定支援',
    icon: '🗄️',
    difficulty: 'intermediate',
    keywords: ['データベース', '経営情報システム', '意思決定支援', 'OR']
  },
  {
    id: 'embedded_iot',
    name: '組込み・IoT',
    category: 'テクノロジー・システム',
    description: '組込みシステム、IoT、ユビキタスコンピューティング',
    icon: '🔧',
    difficulty: 'advanced',
    keywords: ['組込み', 'IoT', 'ユビキタス', 'HCI']
  },
  {
    id: 'web_ui_ux',
    name: 'Webデザイン・UI/UX',
    category: 'クリエイティブ',
    description: 'Webデザイン、UI/UXデザイン、ユーザビリティ',
    icon: '🎨',
    difficulty: 'beginner',
    keywords: ['Webデザイン', 'UI/UX', 'グラフィックデザイン', 'ブランディング']
  },
  {
    id: 'design_visual',
    name: 'デザイン・視覚表現',
    category: 'クリエイティブ',
    description: '視覚デザイン、イラストレーション、感性工学',
    icon: '🎨',
    difficulty: 'beginner',
    keywords: ['視覚デザイン', 'イラストレーション', '感性工学', 'アート']
  },
  {
    id: 'video_animation',
    name: '映像・アニメーション',
    category: 'クリエイティブ',
    description: '映像制作、アニメーション表現、メディア表現',
    icon: '🎬',
    difficulty: 'intermediate',
    keywords: ['映像制作', 'アニメーション', 'メディア表現', '視覚芸術']
  },
  {
    id: 'computer_music',
    name: 'コンピュータ音楽・サウンドアート',
    category: 'クリエイティブ',
    description: 'コンピュータ音楽、サウンドアート、音声情報処理',
    icon: '🎵',
    difficulty: 'intermediate',
    keywords: ['コンピュータ音楽', 'サウンドアート', '音声情報処理']
  },
  {
    id: 'game_esports',
    name: 'ゲーム開発・eスポーツ',
    category: 'エンターテイメント',
    description: 'ゲームプログラミング、eスポーツ、メタバース',
    icon: '🎮',
    difficulty: 'intermediate',
    keywords: ['ゲーム開発', 'eスポーツ', 'メタバース', 'プログラミング']
  },
  {
    id: 'vr_ar_media',
    name: 'VR/AR・メディアアート',
    category: 'エンターテイメント',
    description: 'VR/AR技術、メディアアート、環境認知',
    icon: '🥽',
    difficulty: 'advanced',
    keywords: ['VR', 'AR', 'メディアアート', '環境行動学']
  }
];

// プログラミング言語データ
export const PROGRAMMING_LANGUAGES: ProgrammingLanguage[] = [
  {
    id: 'python',
    name: 'Python',
    category: 'スクリプト言語',
    description: 'データ分析、AI、Web開発に広く使用',
    difficulty: 'beginner',
    icon: '🐍',
    popularity: 95
  },
  {
    id: 'javascript',
    name: 'JavaScript',
    category: 'Web言語',
    description: 'Web開発、フロントエンド・バックエンド',
    difficulty: 'beginner',
    icon: '⚡',
    popularity: 92
  },
  {
    id: 'java',
    name: 'Java',
    category: 'オブジェクト指向',
    description: 'エンタープライズ開発、Android開発',
    difficulty: 'intermediate',
    icon: '☕',
    popularity: 85
  },
  {
    id: 'cpp',
    name: 'C++',
    category: 'システム言語',
    description: 'システム開発、ゲーム開発、高性能計算',
    difficulty: 'advanced',
    icon: '⚙️',
    popularity: 78
  },
  {
    id: 'csharp',
    name: 'C#',
    category: 'オブジェクト指向',
    description: '.NET開発、ゲーム開発（Unity）',
    difficulty: 'intermediate',
    icon: '🔷',
    popularity: 75
  }
];

// 技術フレームワークデータ
export const TECH_FRAMEWORKS: TechFramework[] = [
  {
    id: 'react',
    name: 'React',
    category: 'フロントエンド',
    description: 'モダンなUI開発ライブラリ',
    difficulty: 'intermediate',
    relatedLanguages: ['javascript'],
    icon: '⚛️'
  },
  {
    id: 'django',
    name: 'Django',
    category: 'バックエンド',
    description: 'Pythonの高レベルWebフレームワーク',
    difficulty: 'intermediate',
    relatedLanguages: ['python'],
    icon: '🎸'
  },
  {
    id: 'tensorflow',
    name: 'TensorFlow',
    category: '機械学習',
    description: '機械学習・ディープラーニングフレームワーク',
    difficulty: 'advanced',
    relatedLanguages: ['python'],
    icon: '🧠'
  }
];

// フィールドカテゴリ
export const FIELD_CATEGORIES = [
  'テクノロジー・システム',
  'クリエイティブ',
  'エンターテイメント'
];

// ユーティリティ関数
export const fieldUtils = {
  getFieldsByCategory: (category: string) => {
    return RESEARCH_FIELDS.filter(field => field.category === category);
  },

  getFieldById: (id: string) => {
    return RESEARCH_FIELDS.find(field => field.id === id);
  }
};

// カテゴリ別フィールド取得関数（互換性のためにエクスポート）
export const getFieldsByCategory = (category: string) => {
  return FIELD_CATEGORIES.includes(category) ? [category] : [];
};

// API サービスクラス
class ApiService {
  private baseURL: string;

  constructor() {
    this.baseURL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
  }

  async evaluateLabs(preferences: EvaluationPreferences): Promise<EvaluationResponse> {
    try {
      const response = await axios.post(`${this.baseURL}/api/v1/evaluation/evaluate`, {
        preferences
      });
      return response.data;
    } catch (error) {
      console.error('評価API呼び出しエラー:', error);
      throw new Error('研究室評価の処理中にエラーが発生しました');
    }
  }

  // 新しい互換性メソッド
  async evaluateCompatibility(profile: StudentProfile): Promise<EvaluationResponse> {
    return this.evaluateLabs(profile.evaluation_criteria);
  }

  async optimizeWithGeneticAlgorithm(profile: StudentProfile): Promise<EvaluationResponse> {
    try {
      const response = await axios.post(`${this.baseURL}/api/v1/optimization/genetic`, profile);
      return response.data;
    } catch (error) {
      console.error('遺伝的アルゴリズム最適化エラー:', error);
      throw new Error('遺伝的アルゴリズム最適化の処理中にエラーが発生しました');
    }
  }

  async getDemoProfile(): Promise<StudentProfile> {
    // デモプロフィールを返す
    return {
      evaluation_criteria: {
        research_intensity: 7.0,
        advisor_style: 6.0,
        team_work: 7.0,
        workload: 6.0,
        theory_practice: 7.0,
        research_field_match: 8.0,
        skill_development: 7.0,
        learning_pace: 6.0,
        difficulty_preference: 7.0,
        communication_style: 6.0,
        meeting_frequency: 6.0,
        lab_atmosphere: 7.0,
        innovation_risk: 6.0,
        methodology_preference: 6.0,
        interdisciplinary: 6.0,
        flexibility: 7.0,
        evening_weekend_work: 5.0,
        publication_opportunity: 8.0,
        financial_support: 7.0,
        lab_hierarchy: 6.0,
        core_time_flexibility: 7.0,
      },
      field_interests: {
        "人工知能・機械学習": 8.0,
        "画像・映像処理": 6.0,
        "Webデザイン・UI/UX": 7.0,
        "ゲーム開発・eスポーツ": 5.0,
        "VR/AR・メディアアート": 6.0,
      }
    };
  }

  async getLabData(): Promise<Laboratory[]> {
    try {
      const response = await axios.get(`${this.baseURL}/api/v1/labs`);
      return response.data;
    } catch (error) {
      console.error('研究室データ取得エラー:', error);
      throw new Error('研究室データの取得中にエラーが発生しました');
    }
  }

  async getHealthStatus() {
    try {
      const response = await axios.get(`${this.baseURL}/health`);
      return response.data;
    } catch (error) {
      console.error('ヘルス状態取得エラー:', error);
      throw new Error('サーバーの状態確認中にエラーが発生しました');
    }
  }

  async getSystemStats() {
    try {
      const response = await axios.get(`${this.baseURL}/api/v1/system/stats`);
      return response.data;
    } catch (error) {
      console.error('システム統計取得エラー:', error);
      throw new Error('システム統計の取得中にエラーが発生しました');
    }
  }
}

export const apiService = new ApiService();