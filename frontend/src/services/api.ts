// frontend/src/services/api.ts - 完全修正版（構文エラー解決）
import axios from 'axios';

// ===== 型定義 =====

export interface EvaluationPreferences {
  research_intensity: number;       // 研究強度
  advisor_style: number;           // 指導スタイル
  team_work: number;               // チームワーク
  workload: number;                // ワークロード
  theory_practice: number;         // 理論・実践バランス
  research_field_match: number;    // 研究分野適合性
  skill_development: number;       // スキル開発
  lab_atmosphere: number;          // 研究室雰囲気
  flexibility: number;             // 柔軟性
  publication_opportunity: number; // 論文発表機会
  interdisciplinary: number;       // 学際性
  communication_style: number;     // コミュニケーション
  innovation_risk: number;         // 革新性・リスク許容度
}

export interface LabResult {
  lab_name: string;
  professor_name: string;
  research_area: string;
  final_score: number;
  detailed_scores?: { [key: string]: number };
  explanation?: string;
  suggestions?: string[];
  keywords?: string[];
  metadata?: {
    student_count?: number;
    equipment_level?: number;
    funding_level?: string;
    faculty_type?: string;
  };
}

export interface EvaluationResponse {
  lab_results: LabResult[];
  summary?: {
    total_labs: number;
    avg_score: number;
    max_score: number;
    min_score: number;
  };
  metadata?: {
    evaluation_time: string;
    algorithm_version: string;
  };
}

export interface StudentProfile {
  evaluation_criteria: EvaluationPreferences;
  field_interests: { [key: string]: number };
  profile_name?: string;
}

export interface ResearchField {
  id: string;
  name: string;
  category: string;
  description: string;
  faculty_count: number;
  keywords?: string[];
}

// ===== 定数データ =====

export const RESEARCH_FIELDS: ResearchField[] = [
  // テクノロジー・システム分野
  { id: 'ai_ml', name: '人工知能・機械学習', category: 'テクノロジー・システム', description: 'AI技術、機械学習、データサイエンス', faculty_count: 7, keywords: ['機械学習', 'データ解析', 'AI'] },
  { id: 'image_processing', name: '画像・映像処理', category: 'テクノロジー・システム', description: 'コンピュータビジョン、画像解析、映像技術', faculty_count: 6, keywords: ['画像処理', 'CV', '映像'] },
  { id: 'network_security', name: 'ネットワーク・セキュリティ', category: 'テクノロジー・システム', description: 'ネットワーク技術、情報セキュリティ、通信システム', faculty_count: 3, keywords: ['セキュリティ', 'ネットワーク'] },
  { id: 'database_systems', name: 'データベース・情報システム', category: 'テクノロジー・システム', description: 'データベース設計、情報システム開発', faculty_count: 3, keywords: ['DB', 'システム'] },
  { id: 'embedded_iot', name: '組込み・IoT', category: 'テクノロジー・システム', description: '組込みシステム、IoT、ユビキタス', faculty_count: 2, keywords: ['組込み', 'IoT'] },
  { id: 'education_linguistics', name: '教育・言語学', category: 'テクノロジー・システム', description: '教育システム、言語処理、教育工学', faculty_count: 5, keywords: ['教育', '言語処理'] },
  { id: 'natural_science_math', name: '自然科学・数理', category: 'テクノロジー・システム', description: '数理科学、自然科学シミュレーション', faculty_count: 6, keywords: ['数理', '科学'] },
  { id: 'medical_healthcare', name: '医療情報・ヘルスケア', category: 'テクノロジー・システム', description: '医療情報システム、ヘルスケアIT', faculty_count: 2, keywords: ['医療', 'ヘルスケア'] },
  { id: 'tourism_regional', name: '観光情報・地域システム', category: 'テクノロジー・システム', description: '観光情報システム、地域情報化', faculty_count: 2, keywords: ['観光', '地域'] },
  { id: 'business_decision', name: '経営情報・意思決定支援', category: 'テクノロジー・システム', description: '経営情報システム、意思決定支援', faculty_count: 3, keywords: ['経営', '意思決定'] },
  { id: 'audio_processing', name: '音声・音響情報処理', category: 'テクノロジー・システム', description: '音声処理、音響信号処理', faculty_count: 2, keywords: ['音声', '音響'] },
  { id: 'system_ethics', name: 'システム運用・情報倫理', category: 'テクノロジー・システム', description: 'システム運用管理、情報倫理', faculty_count: 3, keywords: ['運用', '倫理'] },

  // クリエイティブ分野
  { id: 'web_design', name: 'Webデザイン・UI/UX', category: 'クリエイティブ', description: 'Webデザイン、ユーザーインターフェース設計', faculty_count: 4, keywords: ['Web', 'UI/UX'] },
  { id: 'design_visual', name: 'デザイン・視覚表現', category: 'クリエイティブ', description: 'グラフィックデザイン、視覚芸術', faculty_count: 4, keywords: ['デザイン', 'グラフィック'] },
  { id: 'video_animation', name: '映像・アニメーション', category: 'クリエイティブ', description: '映像制作、アニメーション技術', faculty_count: 2, keywords: ['映像', 'アニメ'] },
  { id: 'computer_music', name: 'コンピュータ音楽・サウンドアート', category: 'クリエイティブ', description: '電子音楽、サウンドアート', faculty_count: 2, keywords: ['音楽', 'サウンド'] },

  // エンターテイメント分野
  { id: 'game_esports', name: 'ゲーム開発・eスポーツ', category: 'エンターテイメント', description: 'ゲーム開発、eスポーツ技術', faculty_count: 2, keywords: ['ゲーム', 'eスポーツ'] },
  { id: 'vr_ar_media', name: 'VR/AR・メディアアート', category: 'エンターテイメント', description: 'VR/AR技術、メディアアート', faculty_count: 2, keywords: ['VR', 'AR', 'メディアアート'] },

  // 人文・社会・体育分野
  { id: 'philosophy_humanities', name: '哲学・人文・環境行動学', category: '人文・社会・体育', description: '哲学、人文科学、環境行動学', faculty_count: 2, keywords: ['哲学', '人文', '環境'] },
  { id: 'sports_science', name: 'スポーツ・体育科学', category: '人文・社会・体育', description: 'スポーツ科学、体育工学', faculty_count: 2, keywords: ['スポーツ', '体育'] }
];

export const FIELD_CATEGORIES = [
  'テクノロジー・システム',
  'クリエイティブ',
  'エンターテイメント',
  '人文・社会・体育'
];

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

// ===== ユーティリティ関数 =====

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

// ===== APIサービスクラス =====

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

  // 研究室評価
  async evaluateLabs(evaluationData: any): Promise<EvaluationResponse> {
    try {
      console.log('🚀 研究室評価開始...', evaluationData);

      // データ形式を正規化 - 複数の形式に対応
      let requestData: any;

      if ('student_profile' in evaluationData) {
        // 新形式: {student_profile: {...}}
        requestData = evaluationData;
      } else if ('evaluation_criteria' in evaluationData && 'field_interests' in evaluationData) {
        // 旧形式: {evaluation_criteria: {...}, field_interests: {...}}
        requestData = {
          student_profile: {
            ...evaluationData.evaluation_criteria,
            field_interests: evaluationData.field_interests
          }
        };
      } else if (typeof evaluationData === 'object' && 'research_intensity' in evaluationData) {
        // 直接形式: EvaluationPreferencesオブジェクト
        requestData = {
          student_profile: evaluationData
        };
      } else {
        // フォールバック
        requestData = {
          student_profile: evaluationData
        };
      }

      console.log('📤 送信データ:', requestData);

      const response = await axios.post(`${this.baseURL}/api/evaluate`, requestData, {
        timeout: 30000,
        headers: { 'Content-Type': 'application/json' }
      });

      console.log('📥 レスポンス受信:', response.data);

      // レスポンスデータの正規化
      const data = response.data;
      const normalizedResponse: EvaluationResponse = {
        lab_results: data.lab_results || data.results || [],
        summary: data.summary || {
          total_labs: data.lab_results?.length || 0,
          avg_score: 0,
          max_score: 0,
          min_score: 0
        },
        metadata: data.metadata
      };

      console.log('✅ 評価完了:', normalizedResponse);
      return normalizedResponse;
    } catch (error) {
      console.error('❌ 評価エラー:', error);

      if (axios.isAxiosError(error)) {
        if (error.response?.status === 400) {
          const errorDetail = error.response.data?.detail || '入力データに問題があります';
          throw new Error(`入力エラー: ${errorDetail}`);
        } else if (error.response?.status === 404) {
          throw new Error('APIエンドポイントが見つかりません。バックエンドの実装を確認してください。');
        } else if (error.response?.status === 500) {
          const errorDetail = error.response.data?.detail || 'サーバー内部エラー';
          throw new Error(`サーバーエラー: ${errorDetail}`);
        } else if (error.response?.status === 503) {
          throw new Error('システムが初期化されていません。バックエンドサーバーを再起動してください。');
        }
      }

      throw new Error(`評価処理でエラーが発生しました: ${error}`);
    }
  }

  // 最適化実行
  async runOptimization(preferences: EvaluationPreferences): Promise<EvaluationResponse> {
    try {
      const response = await axios.post(`${this.baseURL}/api/optimize`, {
        evaluation_criteria: preferences
      }, {
        timeout: 60000 // 最適化は時間がかかる可能性があるため長めに設定
      });

      return response.data;
    } catch (error) {
      console.error('最適化エラー:', error);
      throw new Error(`最適化処理でエラーが発生しました: ${error}`);
    }
  }

  // デモプロファイル取得
  async getDemoProfile(): Promise<StudentProfile> {
    try {
      const response = await axios.get(`${this.baseURL}/api/demo-profile`);
      return response.data;
    } catch (error) {
      console.error('デモプロファイル取得エラー:', error);
      // フォールバック用のデモデータ
      return {
        evaluation_criteria: {
          research_intensity: 7,
          advisor_style: 6,
          team_work: 8,
          workload: 6,
          theory_practice: 7,
          research_field_match: 9,
          skill_development: 8,
          lab_atmosphere: 7,
          flexibility: 6,
          publication_opportunity: 8,
          interdisciplinary: 7,
          communication_style: 6,
          innovation_risk: 7,
        },
        field_interests: {
          'ai_ml': 9,
          'image_processing': 7,
          'web_design': 6
        },
        profile_name: 'デモユーザー'
      };
    }
  }

  // 研究分野データ取得
  async getResearchFields(): Promise<{ research_fields: ResearchField[]; total_fields: number; total_labs: number }> {
    try {
      const response = await axios.get(`${this.baseURL}/api/research-fields`);
      return response.data;
    } catch (error) {
      console.error('研究分野取得エラー:', error);
      // フォールバック：ローカルデータを返す
      return {
        research_fields: RESEARCH_FIELDS,
        total_fields: RESEARCH_FIELDS.length,
        total_labs: RESEARCH_FIELDS.reduce((sum, field) => sum + field.faculty_count, 0)
      };
    }
  }
}

// ===== エクスポート =====

export const apiService = new ApiService();

export const testApiConnection = async (): Promise<boolean> => {
  return await apiService.testConnection();
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