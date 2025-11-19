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

  // デザイン分野（4分野）
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
  // 20種類のデモプロファイル
  return [
    'AI研究集中型',
    '実践・就職重視型',
    'バランス型',
    '自由度重視型',
    '理論研究型',
    'チーム協働型',
    '個人研究型',
    '論文発表重視型',
    'スキル開発型',
    '学際的研究型',
    '柔軟スケジュール型',
    '厳格指導型',
    'デザイン実践型',
    'ゲーム開発型',
    'データサイエンス型',
    'Webエンジニア型',
    '映像制作型',
    '教育研究型',
    '軽負荷研究型',
    '企業連携型'
  ];
};

/**
 * デモプロファイルデータ
 */
const DEMO_PROFILES: { [key: string]: EvaluationRequest } = {
  'AI研究集中型': {
    research_intensity: 9, research_intensity_priority: 10,
    advisor_style: 7, advisor_style_priority: 6,
    team_work: 6, team_work_priority: 5,
    workload: 9, workload_priority: 7,
    theory_practice: 5, theory_practice_priority: 6,
    skill_development: 8, skill_development_priority: 8,
    lab_atmosphere: 7, lab_atmosphere_priority: 5,
    flexibility: 5, flexibility_priority: 4,
    publication_opportunity: 10, publication_opportunity_priority: 10,
    research_field_match: 9, research_field_match_priority: 9,
    interdisciplinary: 6, interdisciplinary_priority: 5,
    communication_style: 6, communication_style_priority: 4,
    field_interests: {
      'ai_ml': 10,
      'image_processing': 8,
      'data_science_math': 7,
      'natural_science': 6
    }
  },

  '実践・就職重視型': {
    research_intensity: 5, research_intensity_priority: 5,
    advisor_style: 8, advisor_style_priority: 7,
    team_work: 7, team_work_priority: 6,
    workload: 6, workload_priority: 8,
    theory_practice: 9, theory_practice_priority: 10,
    skill_development: 9, skill_development_priority: 10,
    lab_atmosphere: 8, lab_atmosphere_priority: 7,
    flexibility: 8, flexibility_priority: 9,
    publication_opportunity: 4, publication_opportunity_priority: 3,
    research_field_match: 7, research_field_match_priority: 6,
    interdisciplinary: 7, interdisciplinary_priority: 6,
    communication_style: 8, communication_style_priority: 7,
    field_interests: {
      'software_dev': 10,
      'web_design_uiux': 8,
      'database_systems': 7
    }
  },

  'バランス型': {
    research_intensity: 6, research_intensity_priority: 5,
    advisor_style: 6, advisor_style_priority: 5,
    team_work: 6, team_work_priority: 5,
    workload: 6, workload_priority: 5,
    theory_practice: 6, theory_practice_priority: 5,
    skill_development: 6, skill_development_priority: 5,
    lab_atmosphere: 6, lab_atmosphere_priority: 5,
    flexibility: 6, flexibility_priority: 5,
    publication_opportunity: 6, publication_opportunity_priority: 5,
    research_field_match: 6, research_field_match_priority: 5,
    interdisciplinary: 6, interdisciplinary_priority: 5,
    communication_style: 6, communication_style_priority: 5,
    field_interests: {
      'ai_ml': 7,
      'web_design_uiux': 7,
      'software_dev': 6
    }
  },

  '自由度重視型': {
    research_intensity: 5, research_intensity_priority: 4,
    advisor_style: 9, advisor_style_priority: 10,
    team_work: 4, team_work_priority: 3,
    workload: 4, workload_priority: 6,
    theory_practice: 7, theory_practice_priority: 5,
    skill_development: 7, skill_development_priority: 6,
    lab_atmosphere: 7, lab_atmosphere_priority: 6,
    flexibility: 10, flexibility_priority: 10,
    publication_opportunity: 4, publication_opportunity_priority: 3,
    research_field_match: 6, research_field_match_priority: 5,
    interdisciplinary: 5, interdisciplinary_priority: 4,
    communication_style: 6, communication_style_priority: 5,
    field_interests: {
      'illustration_art': 8,
      'graphic_visual': 7,
      'video_film': 6
    }
  },

  '理論研究型': {
    research_intensity: 8, research_intensity_priority: 9,
    advisor_style: 5, advisor_style_priority: 6,
    team_work: 3, team_work_priority: 4,
    workload: 7, workload_priority: 6,
    theory_practice: 2, theory_practice_priority: 9,
    skill_development: 5, skill_development_priority: 5,
    lab_atmosphere: 4, lab_atmosphere_priority: 4,
    flexibility: 6, flexibility_priority: 5,
    publication_opportunity: 9, publication_opportunity_priority: 10,
    research_field_match: 8, research_field_match_priority: 8,
    interdisciplinary: 4, interdisciplinary_priority: 5,
    communication_style: 4, communication_style_priority: 4,
    field_interests: {
      'data_science_math': 10,
      'natural_science': 8,
      'ai_ml': 7
    }
  },

  'チーム協働型': {
    research_intensity: 7, research_intensity_priority: 6,
    advisor_style: 6, advisor_style_priority: 5,
    team_work: 10, team_work_priority: 10,
    workload: 7, workload_priority: 6,
    theory_practice: 8, theory_practice_priority: 7,
    skill_development: 7, skill_development_priority: 6,
    lab_atmosphere: 9, lab_atmosphere_priority: 9,
    flexibility: 6, flexibility_priority: 5,
    publication_opportunity: 7, publication_opportunity_priority: 6,
    research_field_match: 7, research_field_match_priority: 6,
    interdisciplinary: 8, interdisciplinary_priority: 8,
    communication_style: 10, communication_style_priority: 10,
    field_interests: {
      'game_dev': 9,
      'web_design_uiux': 8,
      'media_art': 7
    }
  },

  '個人研究型': {
    research_intensity: 8, research_intensity_priority: 8,
    advisor_style: 8, advisor_style_priority: 7,
    team_work: 2, team_work_priority: 8,
    workload: 7, workload_priority: 6,
    theory_practice: 6, theory_practice_priority: 5,
    skill_development: 8, skill_development_priority: 7,
    lab_atmosphere: 4, lab_atmosphere_priority: 3,
    flexibility: 8, flexibility_priority: 8,
    publication_opportunity: 6, publication_opportunity_priority: 5,
    research_field_match: 8, research_field_match_priority: 7,
    interdisciplinary: 3, interdisciplinary_priority: 4,
    communication_style: 3, communication_style_priority: 7,
    field_interests: {
      'software_dev': 10,
      'ai_ml': 8,
      'database_systems': 7
    }
  },

  '論文発表重視型': {
    research_intensity: 9, research_intensity_priority: 9,
    advisor_style: 5, advisor_style_priority: 6,
    team_work: 5, team_work_priority: 5,
    workload: 8, workload_priority: 7,
    theory_practice: 4, theory_practice_priority: 6,
    skill_development: 6, skill_development_priority: 5,
    lab_atmosphere: 7, lab_atmosphere_priority: 6,
    flexibility: 4, flexibility_priority: 4,
    publication_opportunity: 10, publication_opportunity_priority: 10,
    research_field_match: 9, research_field_match_priority: 9,
    interdisciplinary: 6, interdisciplinary_priority: 6,
    communication_style: 6, communication_style_priority: 5,
    field_interests: {
      'ai_ml': 9,
      'image_processing': 8,
      'embedded_iot': 7
    }
  },

  'スキル開発型': {
    research_intensity: 7, research_intensity_priority: 6,
    advisor_style: 7, advisor_style_priority: 6,
    team_work: 7, team_work_priority: 6,
    workload: 7, workload_priority: 6,
    theory_practice: 8, theory_practice_priority: 8,
    skill_development: 10, skill_development_priority: 10,
    lab_atmosphere: 8, lab_atmosphere_priority: 7,
    flexibility: 7, flexibility_priority: 7,
    publication_opportunity: 6, publication_opportunity_priority: 5,
    research_field_match: 7, research_field_match_priority: 6,
    interdisciplinary: 8, interdisciplinary_priority: 8,
    communication_style: 7, communication_style_priority: 6,
    field_interests: {
      'software_dev': 9,
      'web_design_uiux': 9,
      'database_systems': 8,
      'network_security': 7
    }
  },

  '学際的研究型': {
    research_intensity: 8, research_intensity_priority: 7,
    advisor_style: 6, advisor_style_priority: 5,
    team_work: 8, team_work_priority: 7,
    workload: 7, workload_priority: 6,
    theory_practice: 6, theory_practice_priority: 6,
    skill_development: 8, skill_development_priority: 7,
    lab_atmosphere: 8, lab_atmosphere_priority: 7,
    flexibility: 7, flexibility_priority: 6,
    publication_opportunity: 8, publication_opportunity_priority: 7,
    research_field_match: 7, research_field_match_priority: 6,
    interdisciplinary: 10, interdisciplinary_priority: 10,
    communication_style: 8, communication_style_priority: 7,
    field_interests: {
      'ai_ml': 8,
      'tourism_regional': 7,
      'media_art': 7,
      'data_science_math': 6
    }
  },

  '柔軟スケジュール型': {
    research_intensity: 5, research_intensity_priority: 5,
    advisor_style: 9, advisor_style_priority: 8,
    team_work: 5, team_work_priority: 4,
    workload: 5, workload_priority: 7,
    theory_practice: 7, theory_practice_priority: 6,
    skill_development: 7, skill_development_priority: 6,
    lab_atmosphere: 7, lab_atmosphere_priority: 6,
    flexibility: 10, flexibility_priority: 10,
    publication_opportunity: 5, publication_opportunity_priority: 4,
    research_field_match: 6, research_field_match_priority: 5,
    interdisciplinary: 6, interdisciplinary_priority: 5,
    communication_style: 6, communication_style_priority: 5,
    field_interests: {
      'web_design_uiux': 8,
      'graphic_visual': 8,
      'software_dev': 7
    }
  },

  '厳格指導型': {
    research_intensity: 8, research_intensity_priority: 8,
    advisor_style: 2, advisor_style_priority: 9,
    team_work: 6, team_work_priority: 5,
    workload: 8, workload_priority: 7,
    theory_practice: 5, theory_practice_priority: 6,
    skill_development: 7, skill_development_priority: 7,
    lab_atmosphere: 5, lab_atmosphere_priority: 4,
    flexibility: 4, flexibility_priority: 3,
    publication_opportunity: 9, publication_opportunity_priority: 9,
    research_field_match: 8, research_field_match_priority: 8,
    interdisciplinary: 5, interdisciplinary_priority: 5,
    communication_style: 5, communication_style_priority: 4,
    field_interests: {
      'ai_ml': 9,
      'image_processing': 8,
      'natural_science': 7
    }
  },

  'デザイン実践型': {
    research_intensity: 6, research_intensity_priority: 5,
    advisor_style: 7, advisor_style_priority: 6,
    team_work: 8, team_work_priority: 7,
    workload: 7, workload_priority: 6,
    theory_practice: 9, theory_practice_priority: 9,
    skill_development: 8, skill_development_priority: 8,
    lab_atmosphere: 9, lab_atmosphere_priority: 8,
    flexibility: 7, flexibility_priority: 7,
    publication_opportunity: 5, publication_opportunity_priority: 4,
    research_field_match: 8, research_field_match_priority: 8,
    interdisciplinary: 7, interdisciplinary_priority: 7,
    communication_style: 8, communication_style_priority: 7,
    field_interests: {
      'graphic_visual': 10,
      'web_design_uiux': 9,
      'illustration_art': 8,
      'design_thinking_marketing': 7
    }
  },

  'ゲーム開発型': {
    research_intensity: 7, research_intensity_priority: 7,
    advisor_style: 7, advisor_style_priority: 6,
    team_work: 8, team_work_priority: 7,
    workload: 8, workload_priority: 7,
    theory_practice: 9, theory_practice_priority: 9,
    skill_development: 9, skill_development_priority: 9,
    lab_atmosphere: 9, lab_atmosphere_priority: 8,
    flexibility: 6, flexibility_priority: 6,
    publication_opportunity: 7, publication_opportunity_priority: 6,
    research_field_match: 9, research_field_match_priority: 9,
    interdisciplinary: 7, interdisciplinary_priority: 6,
    communication_style: 8, communication_style_priority: 7,
    field_interests: {
      'game_dev': 10,
      'ai_ml': 8,
      'vr_ar_metaverse': 8,
      'image_processing': 7
    }
  },

  'データサイエンス型': {
    research_intensity: 8, research_intensity_priority: 8,
    advisor_style: 6, advisor_style_priority: 5,
    team_work: 5, team_work_priority: 5,
    workload: 7, workload_priority: 6,
    theory_practice: 5, theory_practice_priority: 6,
    skill_development: 8, skill_development_priority: 8,
    lab_atmosphere: 6, lab_atmosphere_priority: 5,
    flexibility: 7, flexibility_priority: 6,
    publication_opportunity: 8, publication_opportunity_priority: 8,
    research_field_match: 9, research_field_match_priority: 9,
    interdisciplinary: 7, interdisciplinary_priority: 7,
    communication_style: 6, communication_style_priority: 5,
    field_interests: {
      'data_science_math': 10,
      'ai_ml': 9,
      'database_systems': 8,
      'natural_science': 7
    }
  },

  'Webエンジニア型': {
    research_intensity: 6, research_intensity_priority: 6,
    advisor_style: 7, advisor_style_priority: 6,
    team_work: 7, team_work_priority: 6,
    workload: 7, workload_priority: 7,
    theory_practice: 9, theory_practice_priority: 9,
    skill_development: 9, skill_development_priority: 9,
    lab_atmosphere: 8, lab_atmosphere_priority: 7,
    flexibility: 8, flexibility_priority: 7,
    publication_opportunity: 5, publication_opportunity_priority: 4,
    research_field_match: 8, research_field_match_priority: 8,
    interdisciplinary: 6, interdisciplinary_priority: 6,
    communication_style: 7, communication_style_priority: 6,
    field_interests: {
      'web_design_uiux': 10,
      'software_dev': 9,
      'database_systems': 8,
      'network_security': 7
    }
  },

  '映像制作型': {
    research_intensity: 7, research_intensity_priority: 6,
    advisor_style: 7, advisor_style_priority: 6,
    team_work: 9, team_work_priority: 8,
    workload: 8, workload_priority: 7,
    theory_practice: 9, theory_practice_priority: 9,
    skill_development: 8, skill_development_priority: 7,
    lab_atmosphere: 9, lab_atmosphere_priority: 9,
    flexibility: 6, flexibility_priority: 6,
    publication_opportunity: 6, publication_opportunity_priority: 5,
    research_field_match: 9, research_field_match_priority: 9,
    interdisciplinary: 7, interdisciplinary_priority: 7,
    communication_style: 9, communication_style_priority: 8,
    field_interests: {
      'video_film': 10,
      'animation': 8,
      'media_art': 7,
      'computer_music': 6
    }
  },

  '教育研究型': {
    research_intensity: 6, research_intensity_priority: 6,
    advisor_style: 6, advisor_style_priority: 5,
    team_work: 7, team_work_priority: 6,
    workload: 6, workload_priority: 5,
    theory_practice: 6, theory_practice_priority: 6,
    skill_development: 6, skill_development_priority: 6,
    lab_atmosphere: 7, lab_atmosphere_priority: 7,
    flexibility: 7, flexibility_priority: 7,
    publication_opportunity: 7, publication_opportunity_priority: 6,
    research_field_match: 8, research_field_match_priority: 8,
    interdisciplinary: 8, interdisciplinary_priority: 8,
    communication_style: 8, communication_style_priority: 7,
    field_interests: {
      'japanese_education': 10,
      'educational_tech': 8,
      'korean_studies': 7,
      'english_humanities': 6
    }
  },

  '軽負荷研究型': {
    research_intensity: 4, research_intensity_priority: 6,
    advisor_style: 8, advisor_style_priority: 7,
    team_work: 5, team_work_priority: 4,
    workload: 3, workload_priority: 9,
    theory_practice: 7, theory_practice_priority: 5,
    skill_development: 6, skill_development_priority: 5,
    lab_atmosphere: 7, lab_atmosphere_priority: 6,
    flexibility: 9, flexibility_priority: 9,
    publication_opportunity: 4, publication_opportunity_priority: 3,
    research_field_match: 6, research_field_match_priority: 5,
    interdisciplinary: 5, interdisciplinary_priority: 4,
    communication_style: 6, communication_style_priority: 5,
    field_interests: {
      'english_humanities': 8,
      'japanese_education': 7,
      'sports_science': 6
    }
  },

  '企業連携型': {
    research_intensity: 7, research_intensity_priority: 7,
    advisor_style: 7, advisor_style_priority: 6,
    team_work: 8, team_work_priority: 8,
    workload: 7, workload_priority: 7,
    theory_practice: 9, theory_practice_priority: 9,
    skill_development: 9, skill_development_priority: 9,
    lab_atmosphere: 8, lab_atmosphere_priority: 7,
    flexibility: 6, flexibility_priority: 6,
    publication_opportunity: 6, publication_opportunity_priority: 5,
    research_field_match: 8, research_field_match_priority: 8,
    interdisciplinary: 8, interdisciplinary_priority: 8,
    communication_style: 8, communication_style_priority: 8,
    field_interests: {
      'web_design_uiux': 9,
      'software_dev': 9,
      'tourism_regional': 8,
      'design_thinking_marketing': 8
    }
  }
};

/**
 * デモプロファイルの詳細データを取得
 */
export const getDemoProfileSimple = async (profileName: string): Promise<EvaluationRequest> => {
  // プロファイルデータから取得
  if (DEMO_PROFILES[profileName]) {
    return DEMO_PROFILES[profileName];
  }

  // 見つからない場合はデフォルト値を返す
  console.warn(`プロファイル "${profileName}" が見つかりません。デフォルト値を返します。`);
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
    research_intensity_priority: 5,
    advisor_style_priority: 5,
    team_work_priority: 5,
    workload_priority: 5,
    theory_practice_priority: 5,
    skill_development_priority: 5,
    lab_atmosphere_priority: 5,
    flexibility_priority: 5,
    publication_opportunity_priority: 5,
    research_field_match_priority: 5,
    interdisciplinary_priority: 5,
    communication_style_priority: 5,
    field_interests: {}
  };
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