// frontend/src/services/api.ts - シンプル版
import axios from 'axios';

// 型定義
export interface EvaluationPreferences {
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
  innovation_risk: number;
}

export interface StudentProfile {
  evaluation_criteria: EvaluationPreferences;
  field_interests: { [key: string]: number };
  student_id?: string;
}

export interface Laboratory {
  id: string;
  name: string;
  professor: string;
  research_area: string;
  specialization: string;
  description?: string;
}

export interface LabResult {
  lab: Laboratory;
  lab_name?: string;
  advisor?: string;
  overall_score?: number;
  compatibility?: {
    overall_score: number;
    criterion_scores?: any;
  };
  ranking_position?: number;
  recommendations?: string[];
}

export interface EvaluationResponse {
  lab_results: LabResult[];
  results: LabResult[];
  summary: {
    total_labs: number;
    avg_score: number;
    best_match_lab?: string;
    field_analysis?: any;
    recommendations?: string[];
  };
  metadata?: any;
}

// 研究分野データ（完全版）
export interface ResearchField {
  id: string;
  name: string;
  category: string;
  description: string;
  faculty_count: number;
  popularity: number;
}

export const RESEARCH_FIELDS: ResearchField[] = [
  // テクノロジー・システム分野
  {
    id: 'ai_machine_learning',
    name: '人工知能・機械学習',
    category: 'テクノロジー・システム',
    description: 'AI、機械学習、ディープラーニング、自然言語処理の研究',
    faculty_count: 7,
    popularity: 95
  },
  {
    id: 'image_video_processing',
    name: '画像・映像処理',
    category: 'テクノロジー・システム',
    description: '画像処理、コンピュータビジョン、医用画像処理の研究',
    faculty_count: 6,
    popularity: 85
  },
  {
    id: 'network_security',
    name: 'コンピュータネットワーク・セキュリティ',
    category: 'テクノロジー・システム',
    description: 'ネットワーク技術、情報セキュリティ、暗号化の研究',
    faculty_count: 3,
    popularity: 80
  },
  {
    id: 'database_systems',
    name: 'データベース・情報システム',
    category: 'テクノロジー・システム',
    description: 'データベース技術、情報システム、ビッグデータ処理の研究',
    faculty_count: 3,
    popularity: 75
  },
  {
    id: 'embedded_iot',
    name: '組込み・IoT',
    category: 'テクノロジー・システム',
    description: '組込みシステム、IoT、ユビキタスコンピューティングの研究',
    faculty_count: 2,
    popularity: 70
  },

  // クリエイティブ分野
  {
    id: 'web_ui_ux',
    name: 'Webデザイン・UI/UX',
    category: 'クリエイティブ',
    description: 'Webデザイン、ユーザインタフェース、UX設計の研究',
    faculty_count: 4,
    popularity: 80
  },
  {
    id: 'design_visual',
    name: 'デザイン・視覚表現',
    category: 'クリエイティブ',
    description: 'グラフィックデザイン、視覚デザイン、ブランディングの研究',
    faculty_count: 4,
    popularity: 75
  },
  {
    id: 'video_animation',
    name: '映像・アニメーション',
    category: 'クリエイティブ',
    description: '映像制作、アニメーション表現、メディアアートの研究',
    faculty_count: 2,
    popularity: 70
  },
  {
    id: 'computer_music',
    name: 'コンピュータ音楽・サウンドアート',
    category: 'クリエイティブ',
    description: 'コンピュータ音楽、サウンドデザイン、音響技術の研究',
    faculty_count: 2,
    popularity: 65
  },

  // エンターテイメント分野
  {
    id: 'game_esports',
    name: 'ゲーム開発・eスポーツ',
    category: 'エンターテイメント',
    description: 'ゲームプログラミング、eスポーツ、ゲーミフィケーションの研究',
    faculty_count: 2,
    popularity: 85
  },
  {
    id: 'vr_ar_media_art',
    name: 'VR/AR・メディアアート',
    category: 'エンターテイメント',
    description: 'バーチャルリアリティ、拡張現実、メディアアートの研究',
    faculty_count: 2,
    popularity: 80
  }
];

export const FIELD_CATEGORIES = [
  'テクノロジー・システム',
  'クリエイティブ',
  'エンターテイメント'
];

// 評価基準情報
export const CRITERIA_INFO = {
  research_intensity: {
    name: '研究強度',
    description: '研究にどれだけ集中的に取り組みたいか',
    range: '1(軽い研究) ～ 10(集中研究)'
  },
  advisor_style: {
    name: '指導スタイル',
    description: '教授からの指導の受け方の好み',
    range: '1(厳格指導) ～ 10(自由指導)'
  },
  team_work: {
    name: 'チームワーク',
    description: '研究での他者との協働の程度',
    range: '1(個人研究) ～ 10(チーム研究)'
  },
  workload: {
    name: 'ワークロード',
    description: '研究活動の忙しさに対する許容度',
    range: '1(軽い負荷) ～ 10(重い負荷)'
  },
  theory_practice: {
    name: '理論・実践バランス',
    description: '理論研究と実践的研究のバランス',
    range: '1(理論重視) ～ 10(実践重視)'
  },
  research_field_match: {
    name: '研究分野適合性',
    description: '自分の興味と研究室の分野の一致度',
    range: '1(広い分野) ～ 10(専門特化)'
  },
  skill_development: {
    name: 'スキル開発',
    description: '専門性と汎用性のバランス',
    range: '1(専門特化) ～ 10(幅広いスキル)'
  },
  lab_atmosphere: {
    name: '研究室雰囲気',
    description: '研究室の全体的な雰囲気',
    range: '1(静寂集中) ～ 10(活発議論)'
  },
  flexibility: {
    name: '柔軟性',
    description: '研究時間の自由度',
    range: '1(固定スケジュール) ～ 10(柔軟スケジュール)'
  },
  publication_opportunity: {
    name: '論文発表機会',
    description: '研究成果の論文化機会',
    range: '1(少ない機会) ～ 10(豊富な機会)'
  },
  interdisciplinary: {
    name: '学際性',
    description: '他分野との連携の程度',
    range: '1(単一分野) ～ 10(学際連携)'
  },
  communication_style: {
    name: 'コミュニケーション',
    description: '研究室での交流スタイル',
    range: '1(少人数密接) ～ 10(オープン交流)'
  },
  innovation_risk: {
    name: '革新性・リスク許容度',
    description: '新しい手法への挑戦度',
    range: '1(安全手法) ～ 10(革新手法)'
  }
};

export interface ResearchFieldInterests {
  [key: string]: number;
}

// ユーティリティ関数
export const fieldUtils = {
  getFieldsByCategory: (category: string): ResearchField[] => {
    return RESEARCH_FIELDS.filter(field => field.category === category);
  },

  getFieldById: (id: string): ResearchField | undefined => {
    return RESEARCH_FIELDS.find(field => field.id === id);
  },

  getAllCategories: (): string[] => {
    return FIELD_CATEGORIES;
  }
};

// APIサービスクラス
class ApiService {
  private baseURL: string;

  constructor() {
    this.baseURL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
  }

  // 接続テスト
  async testConnection(): Promise<boolean> {
    try {
      const response = await axios.get(`${this.baseURL}/health`, { timeout: 5000 });
      return response.status === 200;
    } catch (error) {
      console.error('接続テスト失敗:', error);
      return false;
    }
  }

  // 研究室評価（シンプル版）
  async evaluateLabs(preferences: EvaluationPreferences): Promise<EvaluationResponse> {
    try {
      const response = await axios.post(`${this.baseURL}/api/evaluate`, {
        preferences
      }, {
        timeout: 10000
      });

      // レスポンスデータの正規化
      const data = response.data;
      const normalizedResponse: EvaluationResponse = {
        lab_results: data.lab_results || data.results || [],
        results: data.lab_results || data.results || [],
        summary: data.summary || {
          total_labs: 0,
          avg_score: 0,
        },
        metadata: data.metadata
      };

      return normalizedResponse;
    } catch (error) {
      console.error('評価API エラー:', error);
      if (axios.isAxiosError(error)) {
        if (error.code === 'ECONNREFUSED') {
          throw new Error('バックエンドサーバーが起動していません。ターミナルで `cd backend && python app.py` を実行してください。');
        } else if (error.response?.status === 404) {
          throw new Error('APIエンドポイントが見つかりません。バックエンドの実装を確認してください。');
        } else if (error.response?.status === 500) {
          throw new Error(`サーバーエラー: ${error.response.data?.detail || '内部エラー'}`);
        }
      }
      throw new Error('研究室評価の処理中にエラーが発生しました');
    }
  }

  // デモプロフィール取得
  async getDemoProfile(): Promise<StudentProfile> {
    return {
      evaluation_criteria: {
        research_intensity: 7.0,
        advisor_style: 6.0,
        team_work: 7.0,
        workload: 6.0,
        theory_practice: 7.0,
        research_field_match: 8.0,
        skill_development: 7.0,
        lab_atmosphere: 7.0,
        flexibility: 7.0,
        publication_opportunity: 8.0,
        interdisciplinary: 6.0,
        communication_style: 6.0,
        innovation_risk: 6.0,
      },
      field_interests: {
        'ai_machine_learning': 8.0,
        'image_video_processing': 6.0,
        'web_ui_ux': 7.0,
        'game_esports': 5.0,
        'vr_ar_media_art': 6.0,
      }
    };
  }
}

export const apiService = new ApiService();

// 接続テスト関数
export const testApiConnection = async (): Promise<boolean> => {
  return await apiService.testConnection();
};