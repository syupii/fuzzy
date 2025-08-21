import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

// Axiosインスタンス作成
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 10000, // 10秒タイムアウト
});

// 研究分野の型定義
export interface ResearchField {
  id: string;
  name: string;
  description: string;
  category: string;
  keywords: string[];
}

export interface FieldInterest {
  isSelected: boolean;
  interestLevel: number;
}

// 研究分野データ
export const RESEARCH_FIELDS: ResearchField[] = [
  // 情報工学・AI分野
  { id: 'ai', name: '人工知能', description: 'AI、機械学習、深層学習', category: '情報工学', keywords: ['AI', '機械学習', 'ディープラーニング'] },
  { id: 'cv', name: 'コンピュータビジョン', description: '画像認識、映像解析', category: '情報工学', keywords: ['画像処理', '映像認識', 'パターン認識'] },
  { id: 'nlp', name: '自然言語処理', description: 'テキスト解析、言語理解', category: '情報工学', keywords: ['テキストマイニング', '言語モデル', '翻訳'] },
  { id: 'robotics', name: 'ロボティクス', description: 'ロボット工学、制御システム', category: '工学', keywords: ['ロボット', '制御', '自動化'] },
  
  // データサイエンス・分析分野
  { id: 'data_science', name: 'データサイエンス', description: 'ビッグデータ、統計解析', category: 'データ科学', keywords: ['ビッグデータ', '統計', '予測'] },
  { id: 'bioinformatics', name: 'バイオインフォマティクス', description: '生命情報学、ゲノム解析', category: '生命科学', keywords: ['ゲノム', '生命情報', 'バイオ'] },
  
  // エンジニアリング分野
  { id: 'software_engineering', name: 'ソフトウェア工学', description: 'システム設計、開発手法', category: '情報工学', keywords: ['システム開発', 'アーキテクチャ', 'プログラミング'] },
  { id: 'cybersecurity', name: 'サイバーセキュリティ', description: '情報セキュリティ、暗号化', category: '情報工学', keywords: ['セキュリティ', '暗号', 'ネットワーク'] },
  
  // 理論・数学分野
  { id: 'algorithms', name: 'アルゴリズム理論', description: '計算理論、最適化', category: '数学・理論', keywords: ['アルゴリズム', '最適化', '計算量'] },
  { id: 'quantum', name: '量子コンピューティング', description: '量子情報、量子アルゴリズム', category: '物理・量子', keywords: ['量子', '量子ビット', '量子アルゴリズム'] },
];

// 🔥 修正版: 20項目対応のEvaluationPreferences
export interface EvaluationPreferences {
  // 既存項目（5項目）
  research_intensity: number;
  advisor_style: number;
  team_work: number;
  workload: number;
  theory_practice: number;
  
  // 分野適合性（元からの重要項目）
  research_field_match: number;
  
  // 学習・成長関連（3項目）
  skill_development: number;
  learning_pace: number;
  difficulty_preference: number;
  
  // コミュニケーション・環境関連（3項目）
  communication_style: number;
  meeting_frequency: number;
  lab_atmosphere: number;
  
  // 研究アプローチ関連（3項目）
  innovation_risk: number;
  methodology_preference: number;
  interdisciplinary: number;
  
  // 時間・ライフスタイル関連（2項目）
  flexibility: number;
  evening_weekend_work: number;
  
  // 調査結果に基づく追加項目（最優先・4項目）
  publication_opportunity: number;
  financial_support: number;
  lab_hierarchy: number;
  core_time_flexibility: number;
}

// Lab型定義も20項目対応に修正
export interface Lab {
  id: number;
  name: string;
  professor: string;
  research_area: string;
  description: string;
  features: EvaluationPreferences; // 20項目すべて含む
  created_at: string;
}

// 拡張された評価設定（研究分野を含む）
export interface EnhancedEvaluationPreferences extends EvaluationPreferences {
  research_field_interests?: {
    [fieldId: string]: FieldInterest;
  };
}

// フィールドマッチング結果
export interface FieldMatchingResult {
  matched_fields: string[];
  field_scores: { [fieldId: string]: number };
  field_weight: number;
}

// フィールド分析結果
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
  prediction_method?: string; // 'simple_fuzzy' | 'genetic_optimization'
  engine_version?: string;
  field_matching?: FieldMatchingResult;
  decision_path?: any[];
  genetic_info?: {
    individual_id: string;
    generation: number;
    fitness: number;
  };
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
  engine_used?: string;
  evaluation_time?: number;
  field_analysis?: FieldAnalysis;
}

export interface EvaluationResponse {
  results: EvaluationResult[];
  summary: EvaluationSummary;
  algorithm_info: {
    engine?: string;
    current_mode?: string;
    genetic_model_loaded?: boolean;
    criteria_weights?: { [key: string]: number };
  };
}

// フィールド推薦レスポンス
export interface FieldRecommendationResponse {
  recommended_fields: string[];
  confidence_scores: { [fieldId: string]: number };
  reasoning: string;
}

// デモデータレスポンス（20項目対応）
export interface EnhancedDemoDataResponse {
  demo_preferences: EvaluationPreferences; // 20項目すべて含む
  suggested_fields?: string[];
  message: string;
}

// フィールドユーティリティ
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

    // 主要カテゴリーを計算
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

// 🔥 修正版: API関数（20項目対応）
export const apiService = {
  // ヘルスチェック
  async healthCheck() {
    const response = await api.get('/health');
    return response.data;
  },

  // 適合度評価（20項目対応）
  async evaluateCompatibility(preferences: EvaluationPreferences | EnhancedEvaluationPreferences): Promise<EvaluationResponse> {
    const response = await api.post('/evaluate', preferences);
    return response.data;
  },

  // デモデータ取得（20項目対応）
  async getDemoData(): Promise<EnhancedDemoDataResponse> {
    const response = await api.get('/demo-data');
    return response.data;
  },

  // フィールド推薦取得
  async getFieldRecommendations(preferences: Partial<EvaluationPreferences>): Promise<FieldRecommendationResponse> {
    try {
      const response = await api.post('/field-recommendations', preferences);
      return response.data;
    } catch (error) {
      // フォールバック: ランダムな推薦を返す
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

  // エンジン情報取得
  async getEngineInfo() {
    const response = await api.get('/engine/info');
    return response.data;
  },

  // エンジン切り替え
  async switchEngine(mode: 'simple' | 'genetic') {
    const response = await api.post('/engine/switch', { mode });
    return response.data;
  },

  // 遺伝的モデル再読み込み
  async reloadGeneticModel() {
    const response = await api.post('/engine/reload');
    return response.data;
  },

  // 説明付き予測
  async predictWithExplanation(userPreferences: EvaluationPreferences, labFeatures: EvaluationPreferences, modelId?: string) {
    const response = await api.post('/predict/explain', {
      user_preferences: userPreferences,
      lab_features: labFeatures,
      model_id: modelId
    });
    return response.data;
  },

  // 研究室一覧取得
  async getLabs(): Promise<Lab[]> {
    const response = await api.get('/labs');
    return response.data;
  },

  // 研究室詳細取得
  async getLabById(id: number): Promise<Lab> {
    const response = await api.get(`/labs/${id}`);
    return response.data;
  },

  // 評価履歴取得
  async getEvaluationHistory(sessionId?: string) {
    const params = sessionId ? { session_id: sessionId } : {};
    const response = await api.get('/evaluations/history', { params });
    return response.data;
  },

  // 統計情報取得
  async getStatistics() {
    const response = await api.get('/statistics');
    return response.data;
  }
};

// エラーハンドリング
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response) {
      // サーバーエラー
      console.error('API Error:', error.response.data);
    } else if (error.request) {
      // ネットワークエラー
      console.error('Network Error:', error.request);
    } else {
      // その他のエラー
      console.error('Error:', error.message);
    }
    return Promise.reject(error);
  }
);

export default api;