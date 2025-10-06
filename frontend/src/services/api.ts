// frontend/src/services/api.ts
/**
 * API通信サービス v3.0
 * - 12項目評価基準対応
 * - 20研究分野対応
 * - パターンB対応
 */

import axios, { AxiosInstance } from 'axios';

// ===== 型定義 =====

// 学生プロファイル（12項目）
export interface StudentProfile {
  // 基本5項目
  research_intensity: number;
  advisor_style: number;
  team_work: number;
  workload: number;
  theory_practice: number;

  // 拡張5項目
  research_field_match: number;  // 分野重視度（1=基本項目重視, 10=分野重視）
  skill_development: number;
  lab_atmosphere: number;
  flexibility: number;
  publication_opportunity: number;

  // 特殊2項目
  interdisciplinary: number;
  communication_style: number;

  // 優先度（オプショナル）
  research_intensity_priority?: number;
  advisor_style_priority?: number;
  team_work_priority?: number;
  workload_priority?: number;
  theory_practice_priority?: number;
  research_field_match_priority?: number;
  skill_development_priority?: number;
  lab_atmosphere_priority?: number;
  flexibility_priority?: number;
  publication_opportunity_priority?: number;
  interdisciplinary_priority?: number;
  communication_style_priority?: number;

  // 分野興味
  field_interests?: { [key: string]: number };
}

// 研究室情報（12項目）
export interface Laboratory {
  id: string;
  name: string;
  advisor?: string;
  professor?: string;
  field_id: string;
  category?: string;
  research_area?: string;
  description?: string;

  // 基本5項目
  research_intensity: number;
  advisor_style: number;
  team_work: number;
  workload: number;
  theory_practice: number;

  // 拡張5項目
  research_field_match: number;
  skill_development: number;
  lab_atmosphere: number;
  flexibility: number;
  publication_opportunity: number;

  // 特殊2項目
  interdisciplinary: number;
  communication_style: number;
}

// 適合度結果
export interface LabResult {
  lab: Laboratory;
  lab_id?: string;
  lab_name?: string;
  field_name?: string;
  overall_compatibility: number;
  basic_score: number;
  field_score: number;
  field_weight_alpha?: number;
  basic_weight_beta?: number;
  recommendation: string;
  tree_path?: string;
  tree_layers?: string[];
  leaf_criteria?: string[];
  explanation: string;
  criteria_scores?: { [key: string]: number };
  field_detail?: any;
}

// 評価レスポンス
export interface EvaluationResponse {
  evaluation_results: LabResult[];
  summary: {
    total_labs: number;
    avg_score: number;
    best_match?: string;
  };
  system_info?: any;
}

// 研究分野情報
export interface ResearchField {
  id: string;
  name: string;
  category: string;
  description?: string;
  faculty_count: number;
  keywords?: string[];
}

// ===== カテゴリ定義 =====

export const FIELD_CATEGORIES = [
  'テクノロジー・システム',
  'クリエイティブ',
  'エンターテイメント',
  '人文・社会・体育'
] as const;

// ===== 研究分野データ（20分野） =====

export const RESEARCH_FIELDS: ResearchField[] = [
  // テクノロジー・システム分野（12分野）
  {
    id: 'ai_ml',
    name: '人工知能・機械学習',
    category: 'テクノロジー・システム',
    description: 'AI、機械学習、ディープラーニング、自然言語処理の研究',
    faculty_count: 7,
    keywords: ['AI', '機械学習', 'ディープラーニング', 'NLP']
  },
  {
    id: 'image_processing',
    name: '画像・映像処理',
    category: 'テクノロジー・システム',
    description: 'コンピュータビジョン、画像認識、医用画像処理の研究',
    faculty_count: 6,
    keywords: ['画像処理', 'CV', 'パターン認識']
  },
  {
    id: 'network_security',
    name: 'ネットワーク・セキュリティ',
    category: 'テクノロジー・システム',
    description: 'ネットワーク技術、情報セキュリティ、暗号化の研究',
    faculty_count: 3,
    keywords: ['ネットワーク', 'セキュリティ', '暗号']
  },
  {
    id: 'database_systems',
    name: 'データベース・情報システム',
    category: 'テクノロジー・システム',
    description: 'データベース技術、情報システム、ビッグデータ処理の研究',
    faculty_count: 3,
    keywords: ['データベース', '情報システム', 'ビッグデータ']
  },
  {
    id: 'embedded_iot',
    name: '組込み・IoT',
    category: 'テクノロジー・システム',
    description: '組込みシステム、IoT、ユビキタスコンピューティングの研究',
    faculty_count: 2,
    keywords: ['組込み', 'IoT', 'ユビキタス']
  },
  {
    id: 'education_linguistics',
    name: '教育・言語学',
    category: 'テクノロジー・システム',
    description: '教育工学、言語処理、eラーニングシステムの研究',
    faculty_count: 5,
    keywords: ['教育工学', '言語学', 'eラーニング']
  },
  {
    id: 'natural_science_math',
    name: '自然科学・数理',
    category: 'テクノロジー・システム',
    description: '数理科学、シミュレーション、科学計算の研究',
    faculty_count: 6,
    keywords: ['数理科学', 'シミュレーション', '科学計算']
  },
  {
    id: 'tourism_regional',
    name: '観光情報・地域システム',
    category: 'テクノロジー・システム',
    description: '観光情報学、地域活性化、GISの研究',
    faculty_count: 2,
    keywords: ['観光情報', '地域システム', 'GIS']
  },
  {
    id: 'business_decision',
    name: '経営情報・意思決定支援',
    category: 'テクノロジー・システム',
    description: '経営情報システム、意思決定支援、データ分析の研究',
    faculty_count: 3,
    keywords: ['経営情報', '意思決定', 'データ分析']
  },
  {
    id: 'audio_processing',
    name: '音声・音響情報処理',
    category: 'テクノロジー・システム',
    description: '音声認識、音響信号処理、音楽情報処理の研究',
    faculty_count: 2,
    keywords: ['音声処理', '音響', '音楽情報']
  },
  {
    id: 'system_ethics',
    name: 'システム運用・情報倫理',
    category: 'テクノロジー・システム',
    description: 'システム管理、情報倫理、ICT社会論の研究',
    faculty_count: 3,
    keywords: ['システム運用', '情報倫理', 'ICT']
  },
  {
    id: 'medical_healthcare',
    name: '医療情報・ヘルスケア',
    category: 'テクノロジー・システム',
    description: '医療情報システム、ヘルスケアIT、遠隔医療の研究',
    faculty_count: 2,
    keywords: ['医療情報', 'ヘルスケア', '遠隔医療']
  },

  // クリエイティブ分野（4分野）
  {
    id: 'web_design',
    name: 'Webデザイン・UI/UX',
    category: 'クリエイティブ',
    description: 'Webデザイン、ユーザインタフェース、UX設計の研究',
    faculty_count: 4,
    keywords: ['Webデザイン', 'UI/UX', 'インタラクション']
  },
  {
    id: 'design_visual',
    name: 'デザイン・視覚表現',
    category: 'クリエイティブ',
    description: 'グラフィックデザイン、視覚デザイン、ブランディングの研究',
    faculty_count: 4,
    keywords: ['デザイン', '視覚表現', 'グラフィック']
  },
  {
    id: 'video_animation',
    name: '映像・アニメーション',
    category: 'クリエイティブ',
    description: '映像制作、アニメーション表現、メディアアートの研究',
    faculty_count: 2,
    keywords: ['映像', 'アニメーション', 'メディアアート']
  },
  {
    id: 'computer_music',
    name: 'コンピュータ音楽・サウンドアート',
    category: 'クリエイティブ',
    description: 'コンピュータ音楽、サウンドデザイン、音響芸術の研究',
    faculty_count: 2,
    keywords: ['コンピュータ音楽', 'サウンドアート', '音響芸術']
  },

  // エンターテイメント分野（2分野）
  {
    id: 'game_esports',
    name: 'ゲーム開発・eスポーツ',
    category: 'エンターテイメント',
    description: 'ゲーム開発、ゲームデザイン、eスポーツ産業の研究',
    faculty_count: 2,
    keywords: ['ゲーム開発', 'eスポーツ', 'ゲームデザイン']
  },
  {
    id: 'vr_ar_media',
    name: 'VR/AR・メディアアート',
    category: 'エンターテイメント',
    description: '仮想現実、拡張現実、インタラクティブアートの研究',
    faculty_count: 2,
    keywords: ['VR', 'AR', 'メディアアート']
  },

  // 人文・社会・体育分野（2分野）
  {
    id: 'philosophy_humanities',
    name: '哲学・人文・環境行動学',
    category: '人文・社会・体育',
    description: '哲学、人文科学、環境行動学の研究',
    faculty_count: 2,
    keywords: ['哲学', '人文学', '環境行動学']
  },
  {
    id: 'sports_science',
    name: 'スポーツ・体育科学',
    category: '人文・社会・体育',
    description: 'スポーツ科学、体育学、健康科学の研究',
    faculty_count: 2,
    keywords: ['スポーツ科学', '体育学', '健康科学']
  },
];

// ===== 評価基準情報（13項目） =====

export interface CriteriaInfo {
  id: string;
  name: string;
  description: string;
  range: string;
  category: 'basic' | 'extended' | 'special';
}

export const EVALUATION_CRITERIA: CriteriaInfo[] = [
  // 基本項目（5項目）
  {
    id: 'research_intensity',
    name: '研究強度',
    description: '研究にどれだけ集中的に取り組みたいか',
    range: '1（軽い研究）～ 10（集中研究）',
    category: 'basic'
  },
  {
    id: 'advisor_style',
    name: '指導スタイル',
    description: '教授からの指導の受け方の好み',
    range: '1（厳格指導）～ 10（自由指導）',
    category: 'basic'
  },
  {
    id: 'team_work',
    name: 'チームワーク',
    description: '研究での他者との協働の程度',
    range: '1（個人研究）～ 10（チーム研究）',
    category: 'basic'
  },
  {
    id: 'workload',
    name: 'ワークロード',
    description: '研究活動の忙しさに対する許容度',
    range: '1（軽い負荷）～ 10（重い負荷）',
    category: 'basic'
  },
  {
    id: 'theory_practice',
    name: '理論・実践バランス',
    description: '理論研究と実践的研究のバランス',
    range: '1（理論重視）～ 10（実践重視）',
    category: 'basic'
  },

  // 拡張項目（5項目）
  {
    id: 'research_field_match',
    name: '分野重視度',
    description: '分野マッチングと基本項目のどちらに比重を置くか',
    range: '1（基本項目重視）～ 10（分野重視）',
    category: 'extended'
  },
  {
    id: 'skill_development',
    name: 'スキル開発',
    description: '専門性と汎用性のバランス',
    range: '1（専門特化）～ 10（幅広いスキル）',
    category: 'extended'
  },
  {
    id: 'lab_atmosphere',
    name: '研究室雰囲気',
    description: '研究室の全体的な雰囲気',
    range: '1（静寂集中）～ 10（活発議論）',
    category: 'extended'
  },
  {
    id: 'flexibility',
    name: '柔軟性',
    description: '研究時間の自由度',
    range: '1（固定スケジュール）～ 10（柔軟スケジュール）',
    category: 'extended'
  },
  {
    id: 'publication_opportunity',
    name: '論文発表機会',
    description: '研究成果の論文化機会',
    range: '1（少ない機会）～ 10（豊富な機会）',
    category: 'extended'
  },

  // 特殊項目（2項目）
  {
    id: 'interdisciplinary',
    name: '学際性',
    description: '他分野との連携の程度',
    range: '1（単一分野）～ 10（学際連携）',
    category: 'special'
  },
  {
    id: 'communication_style',
    name: 'コミュニケーション',
    description: '研究室での交流スタイル',
    range: '1（少人数密接）～ 10（オープン交流）',
    category: 'special'
  },
];

// ===== APIサービスクラス =====

class ApiService {
  private api: AxiosInstance;
  private baseURL: string;

  constructor() {
    // 環境に応じてベースURLを設定
    this.baseURL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

    this.api = axios.create({
      baseURL: this.baseURL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // レスポンスインターセプター（エラーハンドリング）
    this.api.interceptors.response.use(
      (response) => response,
      (error) => {
        console.error('API Error:', error);
        return Promise.reject(error);
      }
    );
  }

  // 接続テスト
  async testConnection(): Promise<boolean> {
    try {
      const response = await this.api.get('/health');
      return response.status === 200;
    } catch (error) {
      console.error('接続テスト失敗:', error);
      return false;
    }
  }

  // ヘルスチェック
  async getHealth(): Promise<any> {
    try {
      const response = await this.api.get('/health');
      return response.data;
    } catch (error) {
      throw new Error('ヘルスチェックに失敗しました');
    }
  }

  // 研究室一覧取得
  async getLabs(): Promise<Laboratory[]> {
    try {
      const response = await this.api.get('/api/labs');
      return response.data.labs || [];
    } catch (error) {
      console.error('研究室一覧取得エラー:', error);
      throw new Error('研究室一覧の取得に失敗しました');
    }
  }

  // 評価基準一覧取得
  getCriteria(): CriteriaInfo[] {
    return EVALUATION_CRITERIA;
  }

  // 研究分野一覧取得
  getFields(): ResearchField[] {
    return RESEARCH_FIELDS;
  }

  // 適合度評価
  async evaluate(profile: StudentProfile): Promise<EvaluationResponse> {
    try {
      const response = await this.api.post('/api/evaluate', profile);
      const data = response.data;

      // レスポンスの正規化
      const normalizedResponse: EvaluationResponse = {
        evaluation_results: data.evaluation_results || data.lab_results || data.results || [],
        summary: data.summary || {
          total_labs: 0,
          avg_score: 0,
        },
        system_info: data.system_info || data.metadata
      };

      return normalizedResponse;
    } catch (error) {
      console.error('評価API エラー:', error);
      if (axios.isAxiosError(error)) {
        if (error.code === 'ECONNREFUSED') {
          throw new Error('バックエンドサーバーが起動していません');
        } else if (error.response?.status === 404) {
          throw new Error('APIエンドポイントが見つかりません');
        } else if (error.response?.status === 500) {
          throw new Error(`サーバーエラー: ${error.response.data?.detail || '内部エラー'}`);
        }
      }
      throw new Error('研究室評価の処理中にエラーが発生しました');
    }
  }

  // デモプロフィール取得（12項目）
  async getDemoProfile(): Promise<StudentProfile> {
    return {
      // 基本5項目
      research_intensity: 7.0,
      advisor_style: 6.0,
      team_work: 7.0,
      workload: 6.0,
      theory_practice: 7.0,

      // 拡張5項目
      research_field_match: 8.0,  // 分野重視度
      skill_development: 7.0,
      lab_atmosphere: 7.0,
      flexibility: 7.0,
      publication_opportunity: 8.0,

      // 特殊2項目
      interdisciplinary: 6.0,
      communication_style: 6.0,

      // 分野興味
      field_interests: {
        'ai_ml': 9.0,
        'image_processing': 7.0,
        'web_design': 6.0,
      }
    };
  }
}

// シングルトンインスタンス
export const apiService = new ApiService();

// 接続テスト関数
export const testApiConnection = async (): Promise<boolean> => {
  return await apiService.testConnection();
};

// デフォルトエクスポート
export default apiService;