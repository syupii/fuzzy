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

// 研究分野興味度の型定義
export interface ResearchFieldInterest {
  fieldId: string;
  isSelected: boolean;
  interestLevel: number;
}

// 拡張された評価設定
export interface EnhancedEvaluationPreferences {
  // 既存の基本設定
  research_intensity: number;
  advisor_style: number;
  team_work: number;
  workload: number;
  theory_practice: number;
  
  // 新規: 研究分野興味度
  research_field_interests?: {
    [fieldId: string]: {
      isSelected: boolean;
      interestLevel: number;
    };
  };
  
  // メタデータ
  selected_fields_count?: number;
  average_field_interest?: number;
  primary_category?: string;
}

// 後方互換性のため既存の型も保持
export interface EvaluationPreferences {
  research_intensity: number;
  advisor_style: number;
  team_work: number;
  workload: number;
  theory_practice: number;
}

// 研究分野定義
export interface ResearchField {
  id: string;
  name: string;
  description: string;
  category: string;
  icon?: string;
}

// 研究分野データ
export const RESEARCH_FIELDS: ResearchField[] = [
  // AI・データサイエンス系
  { id: 'ai_ml', name: '人工知能・機械学習', description: 'AI、深層学習、パターン認識', category: 'AI・データ' },
  { id: 'data_science', name: 'データサイエンス', description: 'データ解析、統計数理、ビッグデータ', category: 'AI・データ' },
  { id: 'computer_vision', name: 'コンピュータビジョン', description: '画像認識、映像解析、医用画像処理', category: 'AI・データ' },
  
  // メディア・デザイン系
  { id: 'web_design', name: 'Web・UIデザイン', description: 'Webデザイン、ユーザーインターフェース、UX設計', category: 'メディア・デザイン' },
  { id: 'visual_design', name: '視覚デザイン', description: 'グラフィック、イラスト、感性工学、ブランディング', category: 'メディア・デザイン' },
  { id: 'video_animation', name: '映像・アニメーション', description: '映像制作、アニメーション表現、3DCG', category: 'メディア・デザイン' },
  { id: 'media_art', name: 'メディアアート', description: 'デジタルアート、インスタレーション、インタラクティブアート', category: 'メディア・デザイン' },
  
  // エンターテインメント系
  { id: 'game_dev', name: 'ゲーム開発', description: 'ゲームプログラミング、eスポーツ、ゲームデザイン', category: 'エンターテインメント' },
  { id: 'vr_ar', name: 'VR/AR技術', description: '仮想現実、拡張現実、メタバース、空間コンピューティング', category: 'エンターテインメント' },
  { id: 'computer_music', name: 'コンピュータ音楽', description: '音響処理、サウンドアート、音声合成', category: 'エンターテインメント' },
  
  // システム・技術系
  { id: 'network_security', name: 'ネットワーク・セキュリティ', description: '通信システム、情報セキュリティ、暗号技術', category: 'システム・技術' },
  { id: 'system_programming', name: 'システムプログラミング', description: 'ソフトウェア開発、OS、コンパイラ、アーキテクチャ', category: 'システム・技術' },
  { id: 'iot_embedded', name: 'IoT・組み込み', description: 'IoTシステム、組み込み開発、センサーネットワーク', category: 'システム・技術' },
  
  // 応用・学際系
  { id: 'medical_informatics', name: '医療情報学', description: '医療システム、ヘルスケアIT、診療支援AI', category: '応用・学際' },
  { id: 'tourism_informatics', name: '観光情報学', description: '観光システム、地域情報、位置情報サービス', category: '応用・学際' },
  { id: 'educational_tech', name: '教育工学', description: '教育システム、HCI、e-learning、教育データ分析', category: '応用・学際' },
];

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
  };
  // 研究分野対応度（新規追加）
  field_alignment?: {
    [fieldId: string]: number; // 0-10の対応度
  };
  created_at: string;
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
  // 研究分野マッチング結果（新規追加）
  field_matching?: {
    matched_fields: string[];
    field_scores: { [fieldId: string]: number };
    field_weight: number;
  };
  confidence: number;
  weights_used: number[];
  explanation: string;
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
  // 研究分野関連サマリー（新規追加）
  field_analysis?: {
    selected_fields_count: number;
    average_interest: number;
    primary_category: string;
    field_coverage: number; // マッチする研究室の割合
  };
}

export interface EvaluationResponse {
  results: EvaluationResult[];
  summary: EvaluationSummary;
  algorithm_info: {
    engine: string;
    criteria_weights: { [key: string]: number };
    field_weights?: { [fieldId: string]: number }; // 新規追加
  };
}

// API関数
export const apiService = {
  // ヘルスチェック
  async healthCheck() {
    const response = await api.get('/health');
    return response.data;
  },

  // 適合度評価（拡張版）
  async evaluateCompatibility(preferences: EnhancedEvaluationPreferences): Promise<EvaluationResponse> {
    const response = await api.post('/evaluate', preferences);
    return response.data;
  },

  // 後方互換性のため既存API保持
  async evaluateBasicCompatibility(preferences: EvaluationPreferences): Promise<EvaluationResponse> {
    const response = await api.post('/evaluate', preferences);
    return response.data;
  },

  // デモデータ取得（拡張版）
  async getDemoData(): Promise<{ 
    demo_preferences: EnhancedEvaluationPreferences; 
    message: string;
    suggested_fields?: string[];
  }> {
    try {
      const response = await api.get('/demo-data');
      return response.data;
    } catch (error) {
      // バックエンドに対応がない場合はデフォルトを返す
      console.warn('デモデータAPIが利用できません。デフォルト値を使用します。');
      return {
        demo_preferences: {
          research_intensity: 7.0,
          advisor_style: 6.0,
          team_work: 7.0,
          workload: 6.0,
          theory_practice: 7.0,
          research_field_interests: {}
        },
        message: 'デフォルトデータを使用中'
      };
    }
  },

  // 研究分野一覧取得
  async getResearchFields(): Promise<ResearchField[]> {
    try {
      const response = await api.get('/research-fields');
      return response.data.fields || RESEARCH_FIELDS;
    } catch (error) {
      // バックエンドに対応がない場合はフロントエンドの定義を使用
      console.warn('研究分野APIが利用できません。フロントエンドの定義を使用します。');
      return RESEARCH_FIELDS;
    }
  },

  // 研究分野推薦取得
  async getFieldRecommendations(basicPreferences: EvaluationPreferences): Promise<{
    recommended_fields: string[];
    reasons: { [fieldId: string]: string };
  }> {
    try {
      const response = await api.post('/field-recommendations', basicPreferences);
      return response.data;
    } catch (error) {
      // バックエンドに対応がない場合は空の推薦を返す
      console.warn('研究分野推薦APIが利用できません。');
      return { recommended_fields: [], reasons: {} };
    }
  },
};

// ユーティリティ関数
export const fieldUtils = {
  // カテゴリー別にフィールドをグループ化
  groupFieldsByCategory(fields: ResearchField[] = RESEARCH_FIELDS): { [category: string]: ResearchField[] } {
    return fields.reduce((groups, field) => {
      const category = field.category;
      if (!groups[category]) {
        groups[category] = [];
      }
      groups[category].push(field);
      return groups;
    }, {} as { [category: string]: ResearchField[] });
  },

  // 選択されたフィールドから統計を計算
  calculateFieldStats(fieldInterests: { [fieldId: string]: { isSelected: boolean; interestLevel: number } }): {
    selectedCount: number;
    averageInterest: number;
    primaryCategory: string;
    categoryDistribution: { [category: string]: number };
  } {
    const selectedFields = Object.entries(fieldInterests).filter(([_, data]) => data.isSelected);
    const selectedCount = selectedFields.length;
    
    if (selectedCount === 0) {
      return {
        selectedCount: 0,
        averageInterest: 0,
        primaryCategory: '',
        categoryDistribution: {}
      };
    }

    const averageInterest = selectedFields.reduce((sum, [_, data]) => sum + data.interestLevel, 0) / selectedCount;
    
    // カテゴリー分布計算
    const categoryDistribution: { [category: string]: number } = {};
    selectedFields.forEach(([fieldId, _]) => {
      const field = RESEARCH_FIELDS.find(f => f.id === fieldId);
      if (field) {
        categoryDistribution[field.category] = (categoryDistribution[field.category] || 0) + 1;
      }
    });

    // 最多カテゴリー決定
    const primaryCategory = Object.keys(categoryDistribution).reduce((a, b) => 
      categoryDistribution[a] > categoryDistribution[b] ? a : b, ''
    );

    return {
      selectedCount,
      averageInterest,
      primaryCategory,
      categoryDistribution
    };
  },

  // フィールドIDから名前を取得
  getFieldName(fieldId: string): string {
    const field = RESEARCH_FIELDS.find(f => f.id === fieldId);
    return field ? field.name : fieldId;
  },

  // カテゴリーアイコンを取得
  getCategoryIcon(category: string): string {
    const iconMap: { [key: string]: string } = {
      'AI・データ': '🤖',
      'メディア・デザイン': '🎨',
      'エンターテインメント': '🎮',
      'システム・技術': '⚙️',
      '応用・学際': '🏥'
    };
    return iconMap[category] || '📚';
  }
};

export default api;