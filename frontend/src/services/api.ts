// frontend/src/services/api.ts - LabResultの型定義を拡張
import axios, { AxiosInstance } from 'axios';

// API設定
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// ★★★ 修正点1: StudentProfileの型定義を最新化 ★★★
export interface StudentProfile {
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
  field_interests: Record<string, number>;
  research_intensity_priority?: number;
  advisor_style_priority?: number;
  team_work_priority?: number;
  workload_priority?: number;
  theory_practice_priority?: number;
  research_field_match_priority?: number; // `research_field_match`の優先度を追加
  skill_development_priority?: number;
  lab_atmosphere_priority?: number;
  flexibility_priority?: number;
  publication_opportunity_priority?: number;
  interdisciplinary_priority?: number;
  communication_style_priority?: number;
}

// ★★★ 修正点2: LabResultの型定義を拡張 ★★★
export interface LabResult {
  lab_id: string;
  lab_name: string;
  advisor?: string;
  professor_name?: string;
  research_area?: string;
  category?: string;
  overall_compatibility?: number;
  final_score?: number;
  priority_adjusted_score?: number;
  basic_score?: number;
  field_score?: number;
  recommendation?: string;
  explanation?: string;
  feature_scores?: Record<string, number>;
  confidence?: number;
  priority_analysis?: {
    high_priority_match: number;
    medium_priority_match: number;
    low_priority_match: number;
  };
  ai_scores?: {
    fuzzy: number;
    genetic: number;
  };
}

export interface EvaluationSummary {
  total_labs: number;
  avg_score: number;
  avg_compatibility?: number;
  best_match_lab?: string;
  best_match_score?: number;
  high_compatibility_count?: number;
}

export interface EvaluationResponse {
  evaluation_results?: LabResult[];
  lab_results?: LabResult[];
  results?: LabResult[];
  summary?: EvaluationSummary;
  system_info?: any;
  metadata?: any;
  total_labs_evaluated?: number;
}

class ApiService {
  private api: AxiosInstance;

  constructor() {
    this.api = axios.create({
      baseURL: API_BASE_URL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.api.interceptors.request.use(
      (config) => {
        console.log('🚀 API リクエスト:', config.method?.toUpperCase(), config.url);
        if (config.data) console.log('📤 送信データ:', config.data);
        return config;
      },
      (error) => {
        console.error('❌ リクエストエラー:', error);
        return Promise.reject(error);
      }
    );

    this.api.interceptors.response.use(
      (response) => {
        console.log('✅ API レスポンス:', response.status);
        console.log('📥 受信データ:', response.data);
        return response;
      },
      (error) => {
        console.error('❌ レスポンスエラー:', error.response?.status, error.response?.data);
        return Promise.reject(error);
      }
    );
  }

  async evaluate(profile: any): Promise<EvaluationResponse> { // profileの型をanyに緩和
    try {
      const response = await this.api.post('/api/evaluate', { student_profile: profile });
      return response.data;
    } catch (error) {
      console.error('❌ 評価API エラー:', error);
      if (axios.isAxiosError(error)) {
        if (error.code === 'ECONNREFUSED') {
          throw new Error('バックエンドサーバーに接続できません。');
        } else if (error.response) {
          throw new Error(`サーバーエラー: ${error.response.status} ${error.response.data?.detail || '内部エラー'}`);
        }
      }
      throw new Error('研究室評価の処理中に不明なエラーが発生しました。');
    }
  }
}

export const apiService = new ApiService();
export default apiService;