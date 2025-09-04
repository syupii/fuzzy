// frontend/src/services/api.ts - 完全版（不足している関数を追加）

import axios from 'axios';

// バックエンドのURL設定
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Axiosインスタンス作成
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000,
});

// レスポンスインターセプター
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

// 研究分野の型定義
export interface ResearchField {
  id: string;
  name: string;
  description: string;
  category: string;
  keywords: string[];
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  marketDemand: 'high' | 'medium' | 'low';
}

// 分野の興味度設定
export interface FieldInterest {
  isSelected: boolean;
  interestLevel: number;
  priority: 'high' | 'medium' | 'low';
}

// プログラミング言語の型定義
export interface ProgrammingLanguage {
  id: string;
  name: string;
  description: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  marketDemand: 'high' | 'medium' | 'low';
  applications: string[];
  icon: string;
  category: string;
  learningCurve: number;
}

// 技術フレームワークの型定義
export interface TechFramework {
  id: string;
  name: string;
  description: string;
  category: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  icon: string;
  language?: string;
  popularity: number;
  learningResources: 'abundant' | 'moderate' | 'limited';
}

// 技術スタック設定の型定義
export interface TechStackPreference {
  languagePreferences: string[];
  frameworkExperience: string[];
  experienceLevel: 'beginner' | 'intermediate' | 'advanced';
  learningPreference: 'practical' | 'theoretical' | 'mixed';
  learningWillingness: number;
  careerGoals: string[];
}

// 基本的な評価設定
export interface EvaluationPreferences {
  research_intensity: number;
  advisor_style: number;
  team_work: number;
  workload: number;
  theory_practice: number;
}

// 拡張された評価設定
export interface EnhancedEvaluationPreferences extends EvaluationPreferences {
  research_fields?: {
    [fieldId: string]: FieldInterest;
  };
  tech_stack_preferences?: TechStackPreference;
}

// 研究室情報
export interface Lab {
  id: string;
  name: string;
  advisor: string;
  description: string;
  research_intensity: number;
  advisor_style: number;
  team_work: number;
  workload: number;
  theory_practice: number;
  fields: string[];
  publications: number;
  funding: string;
  equipment: string;
  graduate_employment: string;
}

// 評価結果
export interface EvaluationResult {
  lab_id: string;
  lab_name: string;
  advisor: string;
  overall_compatibility: number;
  feature_scores: {
    [key: string]: number;
  };
  confidence: number;
  recommendation: string;
  explanation: string;
}

// 評価レスポンス
export interface EvaluationResponse {
  student_profile: EvaluationPreferences;
  evaluation_results: EvaluationResult[];
  total_labs_evaluated: number;
  evaluation_timestamp: number;
  system_info: {
    fuzzy_enabled: boolean;
    genetic_enabled: boolean;
    evaluation_count: number;
  };
}

// デモデータレスポンス
export interface EnhancedDemoDataResponse {
  demo_preferences: EnhancedEvaluationPreferences;
  suggested_fields?: string[];
  message: string;
}

// 分野カテゴリー（エクスポート追加）
export const FIELD_CATEGORIES = [
  '情報工学',
  'データ科学',
  '生命科学',
  '工学',
  '数学・理論',
  '物理・量子'
];

// 研究分野データ
export const RESEARCH_FIELDS: ResearchField[] = [
  {
    id: 'ai',
    name: '人工知能',
    description: 'AI、機械学習、深層学習',
    category: '情報工学',
    keywords: ['AI', '機械学習', 'ディープラーニング'],
    difficulty: 'advanced',
    marketDemand: 'high'
  },
  {
    id: 'cv',
    name: 'コンピュータビジョン',
    description: '画像認識、映像解析',
    category: '情報工学',
    keywords: ['画像処理', '映像認識', 'パターン認識'],
    difficulty: 'advanced',
    marketDemand: 'high'
  },
  {
    id: 'nlp',
    name: '自然言語処理',
    description: 'テキスト解析、言語理解',
    category: '情報工学',
    keywords: ['テキストマイニング', '言語モデル', '翻訳'],
    difficulty: 'advanced',
    marketDemand: 'high'
  },
  {
    id: 'robotics',
    name: 'ロボティクス',
    description: 'ロボット工学、制御システム',
    category: '工学',
    keywords: ['ロボット', '制御', '自動化'],
    difficulty: 'intermediate',
    marketDemand: 'medium'
  },
  {
    id: 'data_science',
    name: 'データサイエンス',
    description: 'ビッグデータ、統計解析',
    category: 'データ科学',
    keywords: ['ビッグデータ', '統計', '予測'],
    difficulty: 'intermediate',
    marketDemand: 'high'
  },
  {
    id: 'cybersecurity',
    name: 'サイバーセキュリティ',
    description: '情報セキュリティ、暗号化',
    category: '情報工学',
    keywords: ['セキュリティ', '暗号', 'ネットワーク'],
    difficulty: 'advanced',
    marketDemand: 'high'
  },
  {
    id: 'web_development',
    name: 'Web開発',
    description: 'Webアプリケーション、フロントエンド',
    category: '情報工学',
    keywords: ['Web', 'フロントエンド', 'バックエンド'],
    difficulty: 'beginner',
    marketDemand: 'high'
  },
  {
    id: 'mobile_development',
    name: 'モバイル開発',
    description: 'スマートフォンアプリ開発',
    category: '情報工学',
    keywords: ['iOS', 'Android', 'アプリ'],
    difficulty: 'intermediate',
    marketDemand: 'high'
  },
  {
    id: 'bioinformatics',
    name: 'バイオインフォマティクス',
    description: '生命情報学、ゲノム解析',
    category: '生命科学',
    keywords: ['ゲノム', '生命情報', 'バイオ'],
    difficulty: 'advanced',
    marketDemand: 'medium'
  },
  {
    id: 'algorithms',
    name: 'アルゴリズム理論',
    description: '計算理論、最適化',
    category: '数学・理論',
    keywords: ['アルゴリズム', '最適化', '計算量'],
    difficulty: 'advanced',
    marketDemand: 'medium'
  }
];

// プログラミング言語データ
export const PROGRAMMING_LANGUAGES: ProgrammingLanguage[] = [
  {
    id: 'python',
    name: 'Python',
    description: '汎用性が高く、AI・データサイエンス分野で広く使用',
    difficulty: 'beginner',
    marketDemand: 'high',
    applications: ['AI・機械学習', 'データ分析', 'Webアプリ開発'],
    icon: '🐍',
    category: 'スクリプト言語',
    learningCurve: 3
  },
  {
    id: 'javascript',
    name: 'JavaScript',
    description: 'Web開発の基本言語、フロントエンドからバックエンドまで',
    difficulty: 'beginner',
    marketDemand: 'high',
    applications: ['Web開発', 'モバイルアプリ', 'サーバーサイド'],
    icon: '📜',
    category: 'Web言語',
    learningCurve: 4
  },
  {
    id: 'java',
    name: 'Java',
    description: 'エンタープライズシステムで広く使用される安定した言語',
    difficulty: 'intermediate',
    marketDemand: 'high',
    applications: ['企業システム', 'Androidアプリ', 'Webアプリ'],
    icon: '☕',
    category: 'オブジェクト指向言語',
    learningCurve: 6
  },
  {
    id: 'cpp',
    name: 'C++',
    description: '高性能が求められるシステムやゲーム開発に使用',
    difficulty: 'advanced',
    marketDemand: 'medium',
    applications: ['システム開発', 'ゲーム開発', '組み込みシステム'],
    icon: '⚡',
    category: 'システム言語',
    learningCurve: 8
  },
  {
    id: 'golang',
    name: 'Go',
    description: 'Googleが開発した高速でシンプルな言語',
    difficulty: 'intermediate',
    marketDemand: 'medium',
    applications: ['サーバーサイド', 'クラウドサービス', 'マイクロサービス'],
    icon: '🐹',
    category: 'モダン言語',
    learningCurve: 5
  }
];

// 技術フレームワークデータ
export const TECH_FRAMEWORKS: TechFramework[] = [
  {
    id: 'react',
    name: 'React',
    description: 'Facebookが開発したUIライブラリ',
    category: 'フロントエンド',
    difficulty: 'intermediate',
    icon: '⚛️',
    language: 'JavaScript',
    popularity: 9,
    learningResources: 'abundant'
  },
  {
    id: 'vue',
    name: 'Vue.js',
    description: 'プログレッシブなJavaScriptフレームワーク',
    category: 'フロントエンド',
    difficulty: 'beginner',
    icon: '💚',
    language: 'JavaScript',
    popularity: 8,
    learningResources: 'abundant'
  },
  {
    id: 'angular',
    name: 'Angular',
    description: 'Googleが開発したTypeScriptベースのフレームワーク',
    category: 'フロントエンド',
    difficulty: 'advanced',
    icon: '🅰️',
    language: 'TypeScript',
    popularity: 7,
    learningResources: 'abundant'
  },
  {
    id: 'django',
    name: 'Django',
    description: 'Pythonの高レベルWebフレームワーク',
    category: 'バックエンド',
    difficulty: 'intermediate',
    icon: '🎸',
    language: 'Python',
    popularity: 8,
    learningResources: 'abundant'
  },
  {
    id: 'flask',
    name: 'Flask',
    description: '軽量でシンプルなPythonWebフレームワーク',
    category: 'バックエンド',
    difficulty: 'beginner',
    icon: '🌶️',
    language: 'Python',
    popularity: 7,
    learningResources: 'abundant'
  }
];

// ユーティリティ関数（エクスポート追加）
export const fieldUtils = {
  getFieldsByCategory: (category: string): ResearchField[] => {
    return RESEARCH_FIELDS.filter(field => field.category === category);
  },

  getDifficultyColor: (difficulty: 'beginner' | 'intermediate' | 'advanced') => {
    switch (difficulty) {
      case 'beginner': return 'success';
      case 'intermediate': return 'warning';
      case 'advanced': return 'error';
      default: return 'default';
    }
  },

  getMarketDemandColor: (demand: 'high' | 'medium' | 'low') => {
    switch (demand) {
      case 'high': return '#4caf50';
      case 'medium': return '#ff9800';
      case 'low': return '#f44336';
      default: return '#9e9e9e';
    }
  },

  calculateFieldStats: (selectedFields: { [key: string]: FieldInterest }) => {
    const selected = Object.values(selectedFields).filter(f => f.isSelected);
    const selectedCount = selected.length;
    const averageInterest = selected.length > 0
      ? selected.reduce((sum, f) => sum + f.interestLevel, 0) / selected.length
      : 0;
    const highPriorityCount = selected.filter(f => f.priority === 'high').length;

    return { selectedCount, averageInterest, highPriorityCount };
  }
};

// API関数
export const apiService = {
  // ヘルスチェック
  async healthCheck() {
    try {
      const response = await api.get('/health');
      return response.data;
    } catch (error) {
      console.error('Health check failed:', error);
      throw error;
    }
  },

  // 研究室一覧取得
  async getLabs() {
    try {
      const response = await api.get('/api/labs');
      return response.data;
    } catch (error) {
      console.error('Failed to get labs:', error);
      throw error;
    }
  },

  // 適合度評価
  async evaluateCompatibility(preferences: EvaluationPreferences): Promise<EvaluationResponse> {
    try {
      const response = await api.post('/api/evaluate', preferences);
      return response.data;
    } catch (error) {
      console.error('Failed to evaluate compatibility:', error);
      throw error;
    }
  },

  // 最適化実行
  async optimize(studentProfiles: EvaluationPreferences[]) {
    try {
      const response = await api.post('/api/optimize', {
        student_profiles: studentProfiles
      });
      return response.data;
    } catch (error) {
      console.error('Failed to optimize:', error);
      throw error;
    }
  },

  // 説明取得
  async explainRecommendation(studentProfile: EvaluationPreferences, labId: string) {
    try {
      const response = await api.post('/api/explain', {
        student_profile: studentProfile,
        lab_id: labId
      });
      return response.data;
    } catch (error) {
      console.error('Failed to get explanation:', error);
      throw error;
    }
  },

  // デモデータ取得（追加）
  async getDemoData(): Promise<EnhancedDemoDataResponse> {
    try {
      // バックエンドにエンドポイントがない場合のフォールバック
      return {
        demo_preferences: {
          research_intensity: 0.7,
          advisor_style: 0.6,
          team_work: 0.8,
          workload: 0.5,
          theory_practice: 0.4,
          research_fields: {
            'ai': { isSelected: true, interestLevel: 9, priority: 'high' },
            'data_science': { isSelected: true, interestLevel: 7, priority: 'medium' },
            'web_development': { isSelected: true, interestLevel: 6, priority: 'low' }
          },
          tech_stack_preferences: {
            languagePreferences: ['python', 'javascript'],
            frameworkExperience: ['react', 'django'],
            experienceLevel: 'intermediate',
            learningPreference: 'mixed',
            learningWillingness: 8,
            careerGoals: ['research', 'industry']
          }
        },
        suggested_fields: ['ai', 'data_science', 'web_development'],
        message: 'デモデータが読み込まれました'
      };
    } catch (error) {
      console.error('Failed to get demo data:', error);
      // フォールバックデータを返す
      return {
        demo_preferences: {
          research_intensity: 0.5,
          advisor_style: 0.5,
          team_work: 0.5,
          workload: 0.5,
          theory_practice: 0.5
        },
        message: 'デフォルトデータが読み込まれました'
      };
    }
  },
};

export default api;