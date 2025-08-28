// src/services/api.ts - 完全版
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

// Axiosインスタンス作成
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 10000,
});

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

// 分野カテゴリー
export const FIELD_CATEGORIES = [
  '情報工学',
  'データ科学', 
  '生命科学',
  '工学',
  '数学・理論',
  '物理・量子'
];

// 研究分野データ（拡張版）
export const RESEARCH_FIELDS: ResearchField[] = [
  // 情報工学・AI分野
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
  
  // データサイエンス・分析分野
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
    id: 'bioinformatics', 
    name: 'バイオインフォマティクス', 
    description: '生命情報学、ゲノム解析', 
    category: '生命科学', 
    keywords: ['ゲノム', '生命情報', 'バイオ'],
    difficulty: 'advanced',
    marketDemand: 'medium'
  },
  
  // エンジニアリング分野
  { 
    id: 'software_engineering', 
    name: 'ソフトウェア工学', 
    description: 'システム設計、開発手法', 
    category: '情報工学', 
    keywords: ['システム開発', 'アーキテクチャ', 'プログラミング'],
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
  
  // 理論・数学分野
  { 
    id: 'algorithms', 
    name: 'アルゴリズム理論', 
    description: '計算理論、最適化', 
    category: '数学・理論', 
    keywords: ['アルゴリズム', '最適化', '計算量'],
    difficulty: 'advanced',
    marketDemand: 'medium'
  },
  { 
    id: 'quantum', 
    name: '量子コンピューティング', 
    description: '量子情報、量子アルゴリズム', 
    category: '物理・量子', 
    keywords: ['量子', '量子ビット', '量子アルゴリズム'],
    difficulty: 'advanced',
    marketDemand: 'low'
  },
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
    applications: ['企業システム', 'Androidアプリ', 'Webシステム'],
    icon: '☕',
    category: 'オブジェクト指向言語',
    learningCurve: 6
  },
  {
    id: 'cpp',
    name: 'C++',
    description: 'システムプログラミングやゲーム開発に使用',
    difficulty: 'advanced',
    marketDemand: 'medium',
    applications: ['システム開発', 'ゲーム開発', '組み込み'],
    icon: '⚙️',
    category: 'システム言語',
    learningCurve: 8
  },
  {
    id: 'r',
    name: 'R',
    description: '統計解析・データサイエンスに特化した言語',
    difficulty: 'intermediate',
    marketDemand: 'medium',
    applications: ['統計解析', 'データ可視化', '学術研究'],
    icon: '📊',
    category: '統計言語',
    learningCurve: 5
  }
];

// 技術フレームワークデータ
export const TECH_FRAMEWORKS: TechFramework[] = [
  {
    id: 'react',
    name: 'React',
    description: 'Meta社開発のUI構築ライブラリ',
    category: 'フロントエンド',
    difficulty: 'intermediate',
    icon: '⚛️',
    language: 'JavaScript',
    popularity: 9,
    learningResources: 'abundant'
  },
  {
    id: 'tensorflow',
    name: 'TensorFlow',
    description: 'Google開発の機械学習フレームワーク',
    category: 'AI・機械学習',
    difficulty: 'advanced',
    icon: '🧠',
    language: 'Python',
    popularity: 8,
    learningResources: 'abundant'
  },
  {
    id: 'django',
    name: 'Django',
    description: 'Python向け高レベルWebフレームワーク',
    category: 'バックエンド',
    difficulty: 'intermediate',
    icon: '🎸',
    language: 'Python',
    popularity: 7,
    learningResources: 'moderate'
  },
  {
    id: 'spring',
    name: 'Spring',
    description: 'Java向けアプリケーションフレームワーク',
    category: 'バックエンド',
    difficulty: 'intermediate',
    icon: '🌱',
    language: 'Java',
    popularity: 8,
    learningResources: 'abundant'
  }
];

// ユーティリティ関数
export const fieldUtils = {
  getFieldName: (fieldId: string): string => {
    const field = RESEARCH_FIELDS.find(f => f.id === fieldId);
    return field ? field.name : fieldId;
  },
  
  getFieldDescription: (fieldId: string): string => {
    const field = RESEARCH_FIELDS.find(f => f.id === fieldId);
    return field ? field.description : '';
  },
  
  getFieldsByCategory: (category: string): ResearchField[] => {
    return RESEARCH_FIELDS.filter(f => f.category === category);
  },
  
  getAllCategories: (): string[] => {
    const categories = RESEARCH_FIELDS.map(f => f.category);
    return Array.from(new Set(categories));
  },

  groupFieldsByCategory: (fields: ResearchField[]): { [category: string]: ResearchField[] } => {
    const grouped: { [category: string]: ResearchField[] } = {};
    fields.forEach(field => {
      if (!grouped[field.category]) {
        grouped[field.category] = [];
      }
      grouped[field.category].push(field);
    });
    return grouped;
  },

  getCategoryIcon: (category: string): string => {
    const iconMap: { [key: string]: string } = {
      '情報工学': '💻',
      'データ科学': '📊',
      '生命科学': '🧬',
      '工学': '⚙️',
      '数学・理論': '🔢',
      '物理・量子': '⚛️'
    };
    return iconMap[category] || '📚';
  },

  calculateFieldStats: (fieldInterests: { [fieldId: string]: FieldInterest }): {
    selectedCount: number;
    averageInterest: number;
    primaryCategory: string;
  } => {
    const selectedFields = Object.entries(fieldInterests)
      .filter(([_, data]) => data.isSelected);
    
    const selectedCount = selectedFields.length;
    
    if (selectedCount === 0) {
      return {
        selectedCount: 0,
        averageInterest: 0,
        primaryCategory: '未設定'
      };
    }

    const averageInterest = selectedFields
      .reduce((sum, [_, data]) => sum + data.interestLevel, 0) / selectedCount;

    const categoryCount: { [category: string]: number } = {};
    selectedFields.forEach(([fieldId, _]) => {
      const field = RESEARCH_FIELDS.find(f => f.id === fieldId);
      if (field) {
        categoryCount[field.category] = (categoryCount[field.category] || 0) + 1;
      }
    });

    const primaryCategory = Object.entries(categoryCount)
      .sort(([,a], [,b]) => b - a)[0]?.[0] || '未設定';

    return {
      selectedCount,
      averageInterest,
      primaryCategory
    };
  }
};

export const languageUtils = {
  getLanguageName: (langId: string): string => {
    const lang = PROGRAMMING_LANGUAGES.find(l => l.id === langId);
    return lang ? lang.name : langId;
  },

  getLanguageIcon: (langId: string): string => {
    const lang = PROGRAMMING_LANGUAGES.find(l => l.id === langId);
    return lang ? lang.icon : '💻';
  },

  getLanguagesByDifficulty: (difficulty: 'beginner' | 'intermediate' | 'advanced'): ProgrammingLanguage[] => {
    return PROGRAMMING_LANGUAGES.filter(lang => lang.difficulty === difficulty);
  }
};

export const techUtils = {
  getFrameworkName: (frameworkId: string): string => {
    const framework = TECH_FRAMEWORKS.find(f => f.id === frameworkId);
    return framework ? framework.name : frameworkId;
  },

  getFrameworksByCategory: (category: string): TechFramework[] => {
    return TECH_FRAMEWORKS.filter(f => f.category === category);
  }
};

// 既存の型定義
export interface Lab {
  id: number;
  name: string;
  professor: string;
  research_area: string;
  description: string;
  features: {
    research_intensity: number;
    advisor_style: number;
    team_work: number;
    workload: number;
    theory_practice: number;
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
  };
  created_at: string;
}

export interface EvaluationPreferences {
  research_intensity: number;
  advisor_style: number;
  team_work: number;
  workload: number;
  theory_practice: number;
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

export interface EnhancedEvaluationPreferences extends EvaluationPreferences {
  research_field_interests?: {
    [fieldId: string]: FieldInterest;
  };
  tech_stack_preferences?: TechStackPreference;
}

export interface FieldMatchingResult {
  matched_fields: string[];
  field_scores: { [fieldId: string]: number };
  field_weight: number;
}

export interface FieldAnalysis {
  selected_fields_count: number;
  average_interest: number;
  primary_category: string;
  field_coverage: number;
}

export interface CompatibilityResult {
  overall_score: number;
  criterion_scores: {
    [key: string]: {
      similarity: number;
      weighted_score: number;
      user_preference: number;
      lab_feature: number;
      weight: number;
    };
  };
  confidence: number;
  weights_used: number[];
  explanation: string;
  field_matching?: FieldMatchingResult;
}

export interface EvaluationResult {
  lab: Lab;
  compatibility: CompatibilityResult;
}

export interface EvaluationSummary {
  total_labs: number;
  best_match: string;
  avg_score: number;
  evaluation_id: number;
  session_id: string;
  field_analysis?: FieldAnalysis;
}

export interface EvaluationResponse {
  results: EvaluationResult[];
  summary: EvaluationSummary;
  algorithm_info: {
    engine: string;
    criteria_weights: { [key: string]: number };
  };
}

export interface FieldRecommendationResponse {
  recommended_fields: string[];
  confidence_scores: { [fieldId: string]: number };
  reasoning: string;
}

export interface EnhancedDemoDataResponse {
  demo_preferences: EnhancedEvaluationPreferences;
  suggested_fields?: string[];
  message: string;
}

// API関数
export const apiService = {
  async healthCheck() {
    const response = await api.get('/health');
    return response.data;
  },

  async evaluateCompatibility(preferences: EvaluationPreferences | EnhancedEvaluationPreferences): Promise<EvaluationResponse> {
    const response = await api.post('/evaluate', preferences);
    return response.data;
  },

  async getDemoData(): Promise<EnhancedDemoDataResponse> {
    const response = await api.get('/demo-data');
    return response.data;
  },

  async getFieldRecommendations(preferences: Partial<EvaluationPreferences>): Promise<FieldRecommendationResponse> {
    try {
      const response = await api.post('/field-recommendations', preferences);
      return response.data;
    } catch (error) {
      const randomFields = RESEARCH_FIELDS
        .sort(() => Math.random() - 0.5)
        .slice(0, 3)
        .map(f => f.id);
      
      return {
        recommended_fields: randomFields,
        confidence_scores: Object.fromEntries(randomFields.map(id => [id, Math.random() * 0.5 + 0.5])),
        reasoning: 'サーバーからの推薦が利用できないため、一般的な推薦を表示しています。'
      };
    }
  },
};

export default api;