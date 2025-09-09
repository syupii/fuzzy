// frontend/src/services/api.ts - 互換性保持版
import axios from 'axios';

// ===== 既存の型定義を維持 =====

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
  lab_id?: string;
  advisor?: string;
  professor?: string;
  research_area?: string;
  overall_score?: number;
  compatibility_score?: number;
  compatibility?: {
    overall_score: number;
    criterion_scores?: any;
  };
  ranking_position?: number;
  recommendations?: string[];
  description?: string;
  specialization?: string;
  metadata?: any;
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

export interface ResearchFieldInterests {
  [key: string]: number;
}

// ===== 既存の研究分野・評価基準定義を維持 =====

export const RESEARCH_FIELDS: ResearchField[] = [
  // テクノロジー・システム分野
  {
    id: 'ai_machine_learning',
    name: '人工知能・機械学習',
    category: 'テクノロジー・システム',
    description: 'データ解析、機械学習、深層学習、自然言語処理など',
    faculty_count: 7
  },
  {
    id: 'image_video_processing',
    name: '画像・映像処理',
    category: 'テクノロジー・システム',
    description: 'コンピュータビジョン、画像認識、医用画像工学など',
    faculty_count: 6
  },
  {
    id: 'network_security',
    name: 'ネットワーク・セキュリティ',
    category: 'テクノロジー・システム',
    description: 'ネットワーク管理、情報セキュリティ、通信システムなど',
    faculty_count: 3
  },
  {
    id: 'database_systems',
    name: 'データベース・情報システム',
    category: 'テクノロジー・システム',
    description: 'データベース技術、経営情報システム、意思決定支援など',
    faculty_count: 3
  },
  {
    id: 'embedded_iot',
    name: '組込み・IoT',
    category: 'テクノロジー・システム',
    description: '組込みシステム、IoT、ユビキタスコンピューティングなど',
    faculty_count: 2
  },

  // クリエイティブ分野
  {
    id: 'web_ui_ux',
    name: 'Webデザイン・UI/UX',
    category: 'クリエイティブ',
    description: 'Webデザイン、UX/UIデザイン、インタフェースデザインなど',
    faculty_count: 4
  },
  {
    id: 'design_visual',
    name: 'デザイン・視覚表現',
    category: 'クリエイティブ',
    description: '視覚デザイン、グラフィックデザイン、感性工学など',
    faculty_count: 4
  },
  {
    id: 'video_animation',
    name: '映像・アニメーション',
    category: 'クリエイティブ',
    description: '映像制作、アニメーション表現、メディア表現など',
    faculty_count: 2
  },
  {
    id: 'computer_music',
    name: 'コンピュータ音楽・サウンドアート',
    category: 'クリエイティブ',
    description: 'コンピュータ音楽、サウンドアート、音声情報処理など',
    faculty_count: 2
  },

  // エンターテイメント分野
  {
    id: 'game_esports',
    name: 'ゲーム開発・eスポーツ',
    category: 'エンターテイメント',
    description: 'ゲームプログラミング、eスポーツ、メタバースなど',
    faculty_count: 2
  },
  {
    id: 'vr_ar_media_art',
    name: 'VR/AR・メディアアート',
    category: 'エンターテイメント',
    description: 'VR/AR技術、3DCG、メディアアート、認知心理学など',
    faculty_count: 2
  },
];

export const FIELD_CATEGORIES = [
  'テクノロジー・システム',
  'クリエイティブ',
  'エンターテイメント'
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
};

// ===== 既存のユーティリティ関数を維持 =====

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

// ===== APIサービスクラス（修正版） =====

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

  // 研究室評価（修正版 - 既存インターフェース維持）
  async evaluateLabs(preferences: EvaluationPreferences): Promise<EvaluationResponse> {
    try {
      console.log('🚀 研究室評価開始...', preferences);

      // ⭐ 重要: backendが期待する形式で送信（student_profileキーを追加）
      const requestData = {
        student_profile: preferences
      };

      console.log('📤 送信データ:', requestData);

      const response = await axios.post(`${this.baseURL}/api/evaluate`, requestData, {
        timeout: 10000,
        headers: {
          'Content-Type': 'application/json'
        }
      });

      console.log('📥 API応答:', response.data);

      // レスポンスデータの正規化（既存フォーマットに合わせる）
      const data = response.data;

      // lab_resultsをLabResult形式に変換
      const normalizedLabResults: LabResult[] = (data.lab_results || []).map((item: any) => ({
        lab: {
          id: item.lab_id || item.id,
          name: item.lab_name || item.name,
          professor: item.professor,
          research_area: item.research_area,
          specialization: item.specialization || '',
          description: item.description || ''
        },
        lab_name: item.lab_name || item.name,
        lab_id: item.lab_id || item.id,
        advisor: item.professor,
        professor: item.professor,
        research_area: item.research_area,
        overall_score: item.compatibility_score || 0,
        compatibility_score: item.compatibility_score || 0,
        compatibility: {
          overall_score: item.compatibility_score || 0
        },
        description: item.description || '',
        specialization: item.specialization || '',
        metadata: item.metadata || {}
      }));

      const normalizedResponse: EvaluationResponse = {
        lab_results: normalizedLabResults,
        results: normalizedLabResults, // 両方に同じデータを設定
        summary: {
          total_labs: data.total_labs_evaluated || normalizedLabResults.length,
          avg_score: normalizedLabResults.length > 0 ?
            normalizedLabResults.reduce((sum, lab) => sum + (lab.overall_score || 0), 0) / normalizedLabResults.length : 0,
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
          throw new Error(`リクエストエラー: ${detail}`);
        }
      }
      throw new Error('研究室評価の処理中にエラーが発生しました');
    }
  }

  // デモプロフィール取得（既存インターフェース維持）
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

// ===== エクスポート（既存と同じ） =====

export const apiService = new ApiService();

// 接続テスト関数
export const testApiConnection = async (): Promise<boolean> => {
  return await apiService.testConnection();
};