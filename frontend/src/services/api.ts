// frontend/src/services/api.ts - 完全な型定義版
import axios from 'axios';

// ===== 完全な型定義（エラー修正版） =====

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
  research_fields?: string[];
  metadata?: any;
  features?: any;
}

// ⭐ 完全なLabResult型定義（エラー修正）
export interface LabResult {
  // 基本情報
  lab?: Laboratory;
  lab_name?: string;
  lab_id?: string;
  advisor?: string;
  professor?: string;
  research_area?: string;
  description?: string;
  specialization?: string;

  // スコア関連
  overall_score?: number;
  compatibility_score?: number;
  compatibility?: {
    overall_score: number;
    criterion_scores?: { [key: string]: number };
  };
  ranking_position?: number;

  // 追加プロパティ（エラー修正）
  research_fields?: string[];      // ⭐ 追加
  strengths?: string[];           // ⭐ 追加
  considerations?: string[];      // ⭐ 追加
  recommendations?: string[];

  // メタデータ
  metadata?: any;
  features?: any;

  // 分析結果
  feature_analysis?: { [key: string]: any };
  decision_path?: string[];
  explanation?: string;
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
  evaluation_id?: string;
  student_profile?: any;
  total_labs_evaluated?: number;
  timestamp?: number;
  processing_time?: number;
  algorithm_info?: any;
}

export interface ResearchField {
  id: string;
  name: string;
  category: string;
  description?: string;
  faculty_count?: number;
}

// ===== 研究分野定義 =====

export const RESEARCH_FIELDS: ResearchField[] = [
  // テクノロジー・システム分野
  { id: 'ai_machine_learning', name: '人工知能・機械学習', category: 'technology' },
  { id: 'image_video_processing', name: '画像・映像処理', category: 'technology' },
  { id: 'network_security', name: 'コンピュータネットワーク・セキュリティ', category: 'technology' },
  { id: 'database_systems', name: 'データベース・情報システム', category: 'technology' },
  { id: 'embedded_iot', name: '組込み・IoT', category: 'technology' },

  // クリエイティブ分野
  { id: 'web_ui_ux', name: 'Webデザイン・UI/UX', category: 'creative' },
  { id: 'design_visual', name: 'デザイン・視覚表現', category: 'creative' },
  { id: 'video_animation', name: '映像・アニメーション', category: 'creative' },
  { id: 'computer_music', name: 'コンピュータ音楽・サウンドアート', category: 'creative' },

  // エンターテイメント分野
  { id: 'game_esports', name: 'ゲーム開発・eスポーツ', category: 'entertainment' },
  { id: 'vr_ar_media_art', name: 'VR/AR・メディアアート', category: 'entertainment' }
];

export const FIELD_CATEGORIES = ['technology', 'creative', 'entertainment'];

// ===== 評価基準情報 =====

export const CRITERIA_INFO = {
  // 基本項目（5項目）
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

  // 拡張項目（5項目）
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

  // 特殊項目（3項目）
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

// ===== APIサービスクラス（型安全版） =====

class ApiService {
  private baseURL: string;

  constructor() {
    this.baseURL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
  }

  // 接続テスト
  async testConnection(): Promise<boolean> {
    try {
      const response = await axios.get(`${this.baseURL}/health`, { timeout: 5000 });
      console.log('✅ Backend接続成功:', response.data);
      return response.status === 200;
    } catch (error) {
      console.error('❌ Backend接続失敗:', error);
      return false;
    }
  }

  // 研究室評価（チェックボックス対応版）
  async evaluateLabs(preferences: EvaluationPreferences | any): Promise<EvaluationResponse> {
    try {
      console.log('🚀 研究室評価開始...', preferences);

      // チェックボックス式フォーム対応
      let studentProfile: any;
      if ('field_interests' in preferences) {
        // 新形式: チェックボックスで選択された分野情報を含む
        studentProfile = {
          ...preferences,
          innovation_risk: preferences.innovation_risk ?? 6.0
        };
        delete studentProfile.field_interests; // APIでは使わないため除去
      } else {
        // 旧形式: 評価基準のみ
        studentProfile = {
          ...preferences,
          innovation_risk: preferences.innovation_risk ?? 6.0
        };
      }

      const requestData = {
        student_profile: studentProfile
      };

      console.log('📤 送信データ:', requestData);

      const response = await axios.post(`${this.baseURL}/api/evaluate`, requestData, {
        headers: {
          'Content-Type': 'application/json',
        },
        timeout: 30000,
      });

      console.log('📥 レスポンス受信:', response.data);

      const data = response.data;

      // ⭐ 型安全なレスポンス正規化（エラー修正）
      const normalizedLabResults: LabResult[] = (data.lab_results || data.results || []).map((lab: any, index: number): LabResult => ({
        ...lab,
        lab_name: lab.lab_name || lab.name,
        lab_id: lab.lab_id || lab.id,
        professor: lab.professor || lab.advisor,
        overall_score: lab.compatibility_score || lab.overall_score || 0,
        ranking_position: index + 1,
        compatibility: {
          overall_score: lab.compatibility_score || lab.overall_score || 0,
          criterion_scores: lab.criterion_scores || {}
        },
        // 不足しているプロパティのデフォルト値
        research_fields: lab.research_fields || [],
        strengths: lab.strengths || [],
        considerations: lab.considerations || [],
        recommendations: lab.recommendations || []
      }));

      const normalizedResponse: EvaluationResponse = {
        lab_results: normalizedLabResults,
        results: normalizedLabResults,
        summary: {
          total_labs: normalizedLabResults.length,
          avg_score: normalizedLabResults.length > 0 ?
            normalizedLabResults.reduce((sum: number, lab: LabResult) => sum + (lab.overall_score || 0), 0) / normalizedLabResults.length : 0,
          best_match_lab: normalizedLabResults.length > 0 ? normalizedLabResults[0].lab_name : undefined,
          recommendations: []
        },
        metadata: data.metadata,
        evaluation_id: data.evaluation_id,
        student_profile: data.student_profile,
        total_labs_evaluated: data.total_labs_evaluated,
        timestamp: data.timestamp,
        processing_time: data.processing_time,
        algorithm_info: data.algorithm_info
      };

      console.log(`✅ 評価完了: ${normalizedResponse.summary.total_labs}件の研究室`);
      return normalizedResponse;
    } catch (error) {
      console.error('💥 評価API エラー:', error);
      if (axios.isAxiosError(error)) {
        if (error.code === 'ECONNREFUSED') {
          throw new Error('バックエンドサーバーが起動していません。ターミナルで `cd backend && python app.py` を実行してください。');
        } else if (error.response?.status === 404) {
          throw new Error('APIエンドポイントが見つかりません。バックエンドの実装を確認してください。');
        } else if (error.response?.status === 500) {
          const detail = error.response.data?.detail || '内部エラー';
          throw new Error(`サーバーエラー: ${detail}`);
        } else if (error.response?.status === 400) {
          const detail = error.response.data?.detail || 'リクエストエラー';
          console.error('🔍 詳細エラー情報:', error.response.data);
          throw new Error(`リクエストエラー: ${detail}`);
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
        innovation_risk: 6.0
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

  // その他のメソッド
  async getLabs(): Promise<Laboratory[]> {
    try {
      const response = await axios.get(`${this.baseURL}/api/labs`);
      return response.data.labs || [];
    } catch (error) {
      console.error('研究室一覧取得エラー:', error);
      return [];
    }
  }

  async getStatistics(): Promise<any> {
    try {
      const response = await axios.get(`${this.baseURL}/api/statistics`);
      return response.data;
    } catch (error) {
      console.error('統計情報取得エラー:', error);
      return {};
    }
  }

  async getResearchFields(): Promise<any> {
    try {
      const response = await axios.get(`${this.baseURL}/api/research-fields`);
      return response.data;
    } catch (error) {
      console.error('研究分野取得エラー:', error);
      return { research_fields: [], total_fields: 0, total_labs: 0 };
    }
  }
}

// ===== エクスポート =====

export const apiService = new ApiService();

export const testApiConnection = async (): Promise<boolean> => {
  return await apiService.testConnection();
};

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

export const validateEvaluationPreferences = (preferences: Partial<EvaluationPreferences>): string[] => {
  const errors: string[] = [];
  const requiredCriteria = Object.keys(CRITERIA_INFO);

  for (const criterion of requiredCriteria) {
    const value = preferences[criterion as keyof EvaluationPreferences];
    if (value === undefined || value === null) {
      errors.push(`${CRITERIA_INFO[criterion as keyof typeof CRITERIA_INFO].name}が未設定です`);
    } else if (value < 1 || value > 10) {
      errors.push(`${CRITERIA_INFO[criterion as keyof typeof CRITERIA_INFO].name}の値が範囲外です (1-10)`);
    }
  }

  return errors;
};

export default apiService;