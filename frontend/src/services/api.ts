// frontend/src/services/api.ts - 27分野体系対応版
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// ==================== 型定義 ====================

export interface EvaluationRequest {
  // 基本項目（5項目）
  research_intensity: number;
  advisor_style: number;
  team_work: number;
  workload: number;
  theory_practice: number;

  // 拡張項目（5項目）
  skill_development: number;
  lab_atmosphere: number;
  flexibility: number;
  publication_opportunity: number;
  research_field_match: number;

  // 特殊項目（2項目）
  interdisciplinary: number;
  communication_style: number;

  // 優先度（1-10）
  research_intensity_priority?: number;
  advisor_style_priority?: number;
  team_work_priority?: number;
  workload_priority?: number;
  theory_practice_priority?: number;
  skill_development_priority?: number;
  lab_atmosphere_priority?: number;
  flexibility_priority?: number;
  publication_opportunity_priority?: number;
  research_field_match_priority?: number;
  interdisciplinary_priority?: number;
  communication_style_priority?: number;

  // 分野興味（field_id: 1-10）
  field_interests?: { [key: string]: number };
}

export interface LabResult {
  lab_id: string;
  lab_name: string;
  professor_name?: string;
  advisor?: string;
  research_area?: string;
  field_name?: string;
  category?: string;
  overall_compatibility?: number;
  final_score?: number;
  basic_score?: number;
  field_score?: number;
  recommendation?: string;
  explanation?: string;
  tree_path?: string;
  confidence?: number;
  priority_adjusted_score?: number;
  feature_scores?: { [key: string]: number };
  priority_analysis?: {
    high_priority_match: number;
    medium_priority_match: number;
    low_priority_match: number;
  };
  description?: string;
  specialization?: string;
  research_fields?: string[];
  metadata?: {
    faculty_count?: number;
    student_count?: number;
    recent_publications?: number;
    funding_level?: string;
    equipment_rating?: number;
    pattern_type?: string;
    pattern_description?: string;
  };
}

export interface EvaluationResponse {
  evaluation_results?: LabResult[];
  lab_results?: LabResult[];
  results?: LabResult[];
  summary?: {
    total_labs: number;
    avg_score: number;
    high_compatibility_count?: number;
  };
  metadata?: {
    processing_time?: number;
    timestamp?: string;
    ai_engines_used?: string[];
  };
  system_info?: any;
  total_labs_evaluated?: number;
}

// ==================== 27分野定義 ====================

export interface ResearchField {
  id: string;
  name: string;
  category: string;
}

export const RESEARCH_FIELDS: ResearchField[] = [
  // 🔧 テクノロジー・システム分野（10分野）
  { id: 'ai_ml', name: '人工知能・機械学習', category: 'テクノロジー・システム' },
  { id: 'image_processing', name: '画像処理・コンピュータビジョン', category: 'テクノロジー・システム' },
  { id: 'cg_graphics', name: '3DCG・グラフィックス', category: 'テクノロジー・システム' },
  { id: 'network_security', name: 'ネットワーク・セキュリティ', category: 'テクノロジー・システム' },
  { id: 'database_systems', name: 'データベース・情報システム', category: 'テクノロジー・システム' },
  { id: 'embedded_iot', name: '組込み・IoT・HCI', category: 'テクノロジー・システム' },
  { id: 'software_dev', name: 'ソフトウェア開発・アプリ開発', category: 'テクノロジー・システム' },
  { id: 'audio_processing', name: '音声・音響情報処理', category: 'テクノロジー・システム' },
  { id: 'data_science_math', name: 'データサイエンス・統計数理', category: 'テクノロジー・システム' },
  { id: 'natural_science', name: '自然科学・地球物理学', category: 'テクノロジー・システム' },

  // 📚 教育・言語・文化分野（4分野）
  { id: 'japanese_education', name: '日本語教育・言語学', category: '教育・言語・文化' },
  { id: 'korean_studies', name: '韓国語・韓国文化研究', category: '教育・言語・文化' },
  { id: 'educational_tech', name: '教育工学・学習支援', category: '教育・言語・文化' },
  { id: 'english_humanities', name: '英語・人文学', category: '教育・言語・文化' },

  // 🌍 観光・地域分野（1分野）
  { id: 'tourism_regional', name: '観光情報・地域システム', category: '観光・地域' },

  // 🎨 デザイン分野（4分野）
  { id: 'web_design_uiux', name: 'Webデザイン・UI/UX', category: 'デザイン' },
  { id: 'graphic_visual', name: 'グラフィック・視覚デザイン', category: 'デザイン' },
  { id: 'illustration_art', name: 'イラストレーション・アート', category: 'デザイン' },
  { id: 'design_thinking_marketing', name: 'デザイン思考・マーケティング', category: 'デザイン' },

  // 🎬 映像・音楽分野（4分野）
  { id: 'video_film', name: '映像制作・映画', category: '映像・音楽' },
  { id: 'animation', name: 'アニメーション', category: '映像・音楽' },
  { id: 'computer_music', name: 'コンピュータ音楽・サウンドアート', category: '映像・音楽' },
  { id: 'media_art', name: 'メディアアート', category: '映像・音楽' },

  // 🎮 ゲーム・エンタメ分野（3分野）
  { id: 'game_dev', name: 'ゲーム開発', category: 'ゲーム・エンタメ' },
  { id: 'esports', name: 'eスポーツ', category: 'ゲーム・エンタメ' },
  { id: 'vr_ar_metaverse', name: 'VR/AR・メタバース', category: 'ゲーム・エンタメ' },

  // 🏃 人文・社会・体育分野（1分野）
  { id: 'sports_science', name: 'スポーツ科学・バイオメカニクス', category: '人文・社会・体育' }
];

// カテゴリ別に分野をグループ化
export const FIELD_CATEGORIES = {
  'テクノロジー・システム': RESEARCH_FIELDS.filter(f => f.category === 'テクノロジー・システム'),
  '教育・言語・文化': RESEARCH_FIELDS.filter(f => f.category === '教育・言語・文化'),
  '観光・地域': RESEARCH_FIELDS.filter(f => f.category === '観光・地域'),
  'デザイン': RESEARCH_FIELDS.filter(f => f.category === 'デザイン'),
  '映像・音楽': RESEARCH_FIELDS.filter(f => f.category === '映像・音楽'),
  'ゲーム・エンタメ': RESEARCH_FIELDS.filter(f => f.category === 'ゲーム・エンタメ'),
  '人文・社会・体育': RESEARCH_FIELDS.filter(f => f.category === '人文・社会・体育')
};

// ==================== 追加の型定義 ====================

// StudentProfileはEvaluationRequestと同じ構造
export type StudentProfile = EvaluationRequest;

export interface FieldInterest {
  field_id: string;
  interest_level: number;
}

// ==================== API関数 ====================

/**
 * 研究室一覧を取得
 */
export const getLabs = async () => {
  const response = await axios.get(`${API_BASE_URL}/api/labs`);
  return response.data;
};

/**
 * 研究室適合度を評価
 */
export const evaluateLabs = async (data: EvaluationRequest): Promise<EvaluationResponse> => {
  const response = await axios.post(`${API_BASE_URL}/api/evaluate`, data);
  return response.data;
};

/**
 * システム情報を取得
 */
export const getSystemInfo = async () => {
  const response = await axios.get(`${API_BASE_URL}/`);
  return response.data;
};

/**
 * ヘルスチェック
 */
export const healthCheck = async () => {
  const response = await axios.get(`${API_BASE_URL}/health`);
  return response.data;
};

/**
 * デモプロファイル名の一覧を取得
 */
export const getDemoProfileNames = async (): Promise<string[]> => {
  try {
    const response = await axios.get(`${API_BASE_URL}/api/demo-profiles`);
    return response.data.profiles || [];
  } catch (error) {
    console.warn('デモプロファイル取得に失敗、デフォルト値を返します', error);
    return ['AI研究志望', '実践重視型', 'バランス型', '自由度重視'];
  }
};

/**
 * デモプロファイルの詳細データを取得
 */
export const getDemoProfileSimple = async (profileName: string): Promise<EvaluationRequest> => {
  try {
    const response = await axios.get(`${API_BASE_URL}/api/demo-profiles/${encodeURIComponent(profileName)}`);
    return response.data;
  } catch (error) {
    console.warn('デモプロファイル詳細取得に失敗、デフォルト値を返します', error);
    // デフォルトのプロファイルを返す
    return {
      research_intensity: 5,
      advisor_style: 5,
      team_work: 5,
      workload: 5,
      theory_practice: 5,
      skill_development: 5,
      lab_atmosphere: 5,
      flexibility: 5,
      publication_opportunity: 5,
      research_field_match: 5,
      interdisciplinary: 5,
      communication_style: 5,
    };
  }
};

/**
 * 評価を実行（evaluateLabsのエイリアス）
 */
export const evaluate = evaluateLabs;

// ==================== ヘルパー関数 ====================

/**
 * 分野IDから分野名を取得
 */
export const getFieldName = (fieldId: string): string => {
  const field = RESEARCH_FIELDS.find(f => f.id === fieldId);
  return field ? field.name : fieldId;
};

/**
 * 分野IDからカテゴリを取得
 */
export const getFieldCategory = (fieldId: string): string => {
  const field = RESEARCH_FIELDS.find(f => f.id === fieldId);
  return field ? field.category : '不明';
};

/**
 * カテゴリに属する分野を取得
 */
export const getFieldsByCategory = (category: string): ResearchField[] => {
  return RESEARCH_FIELDS.filter(f => f.category === category);
};

export default {
  getLabs,
  evaluateLabs,
  evaluate,
  getSystemInfo,
  healthCheck,
  getDemoProfileNames,
  getDemoProfileSimple,
  getFieldName,
  getFieldCategory,
  getFieldsByCategory,
  RESEARCH_FIELDS,
  FIELD_CATEGORIES
};

// apiServiceという名前でも同じものをエクスポート
export const apiService = {
  getLabs,
  evaluateLabs,
  evaluate,
  getSystemInfo,
  healthCheck,
  getDemoProfileNames,
  getDemoProfileSimple,
  getFieldName,
  getFieldCategory,
  getFieldsByCategory,
  RESEARCH_FIELDS,
  FIELD_CATEGORIES
};