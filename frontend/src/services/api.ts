// frontend/src/services/api.ts - 完全版（型エラー修正）
import axios, { AxiosInstance } from 'axios';

// API設定
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// ========================================
// 基本型定義
// ========================================

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

  // すべて必須フィールドに変更
  research_intensity_priority: number;
  advisor_style_priority: number;
  team_work_priority: number;
  workload_priority: number;
  theory_practice_priority: number;
  research_field_match_priority: number;
  skill_development_priority: number;
  lab_atmosphere_priority: number;
  flexibility_priority: number;
  publication_opportunity_priority: number;
  interdisciplinary_priority: number;
  communication_style_priority: number;

  field_interests: FieldInterest[];
}

export interface FieldInterest {
  field_id: string;
  interest_level: number;
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

// ========================================
// デモプロファイル関連の型定義
// ========================================

export interface DemoProfileInfo {
  description: string;
  characteristics: string[];
}

export interface DemoProfileWithMetadata {
  name: string;
  description: string;
  characteristics: string[];
  profile: StudentProfile;
}

export interface DemoProfilesResponse {
  profiles: {
    [key: string]: DemoProfileInfo;
  };
  count: number;
  message: string;
}

export interface DemoProfileNamesResponse {
  names: string[];
  count: number;
  message: string;
}

export interface DemoStatsResponse {
  total_profiles: number;
  profile_types: {
    [key: string]: number;
  };
  message: string;
}

// ========================================
// APIサービスクラス
// ========================================

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

    // リクエストインターセプター
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

    // レスポンスインターセプター
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

  // ========================================
  // 基本API
  // ========================================

  /**
   * 研究室評価を実行
   * 
   * @param profile - 学生プロファイル
   * @returns 評価結果
   * @throws エラーメッセージ
   */
  async evaluate(profile: any): Promise<EvaluationResponse> {
    try {
      const response = await this.api.post('/api/evaluate', {
        student_profile: profile
      });
      return response.data;
    } catch (error) {
      console.error('❌ 評価API エラー:', error);

      if (axios.isAxiosError(error)) {
        if (error.code === 'ECONNREFUSED') {
          throw new Error('バックエンドサーバーに接続できません。サーバーが起動しているか確認してください。');
        } else if (error.response) {
          const status = error.response.status;
          const detail = error.response.data?.detail || '内部エラー';
          throw new Error(`サーバーエラー (${status}): ${detail}`);
        }
      }

      throw new Error('研究室評価の処理中に不明なエラーが発生しました。');
    }
  }

  /**
   * ヘルスチェック
   * 
   * @returns サーバーが正常に動作している場合true
   */
  async healthCheck(): Promise<boolean> {
    try {
      const response = await this.api.get('/health', { timeout: 5000 });
      return response.status === 200;
    } catch (error) {
      console.error('ヘルスチェック失敗:', error);
      return false;
    }
  }

  /**
   * 研究室一覧を取得
   * 
   * @returns 研究室一覧
   */
  async getLabs(): Promise<Laboratory[]> {
    try {
      const response = await this.api.get('/api/labs');
      return response.data.labs || [];
    } catch (error) {
      console.error('研究室一覧取得エラー:', error);
      throw new Error('研究室一覧の取得に失敗しました');
    }
  }

  // ========================================
  // デモプロファイルAPI
  // ========================================

  /**
   * デモプロファイル一覧を取得
   * 
   * @returns デモプロファイル一覧（説明と特徴のみ）
   * @throws エラーメッセージ
   */
  async getDemoProfiles(): Promise<DemoProfilesResponse> {
    try {
      const response = await this.api.get('/api/demo/profiles');
      return response.data;
    } catch (error) {
      console.error('デモプロファイル一覧の取得に失敗:', error);

      if (axios.isAxiosError(error)) {
        if (error.code === 'ECONNREFUSED') {
          throw new Error('バックエンドサーバーに接続できません。');
        } else if (error.response) {
          throw new Error(`HTTPエラー: ${error.response.status}`);
        }
      }

      throw new Error('デモプロファイル一覧の取得に失敗しました');
    }
  }

  /**
   * デモプロファイル名一覧を取得
   * 
   * @returns プロファイル名の配列
   * @throws エラーメッセージ
   */
  async getDemoProfileNames(): Promise<string[]> {
    try {
      const response = await this.api.get('/api/demo/profiles/names');
      const data: DemoProfileNamesResponse = response.data;
      return data.names;
    } catch (error) {
      console.error('デモプロファイル名一覧の取得に失敗:', error);

      if (axios.isAxiosError(error)) {
        if (error.code === 'ECONNREFUSED') {
          throw new Error('バックエンドサーバーに接続できません。');
        } else if (error.response) {
          throw new Error(`HTTPエラー: ${error.response.status}`);
        }
      }

      throw new Error('デモプロファイル名一覧の取得に失敗しました');
    }
  }

  /**
   * 特定のデモプロファイルを取得（メタデータ付き）
   * 
   * @param profileName - プロファイル名
   * @returns メタデータを含むプロファイルデータ
   * @throws エラーメッセージ
   */
  async getDemoProfile(profileName: string): Promise<DemoProfileWithMetadata> {
    try {
      const encodedName = encodeURIComponent(profileName);
      const response = await this.api.get(`/api/demo/profiles/${encodedName}`);
      return response.data;
    } catch (error) {
      console.error(`デモプロファイル '${profileName}' の取得に失敗:`, error);

      if (axios.isAxiosError(error)) {
        if (error.response?.status === 404) {
          throw new Error(`プロファイル '${profileName}' が見つかりません`);
        } else if (error.code === 'ECONNREFUSED') {
          throw new Error('バックエンドサーバーに接続できません。');
        } else if (error.response) {
          throw new Error(`HTTPエラー: ${error.response.status}`);
        }
      }

      throw new Error(`デモプロファイル '${profileName}' の取得に失敗しました`);
    }
  }

  /**
   * 特定のデモプロファイルを取得（プロファイルデータのみ）
   * 
   * @param profileName - プロファイル名
   * @returns プロファイルデータ（StudentProfile形式）
   * @throws エラーメッセージ
   */
  async getDemoProfileSimple(profileName: string): Promise<StudentProfile> {
    try {
      const encodedName = encodeURIComponent(profileName);
      const response = await this.api.get(`/api/demo/profiles/${encodedName}/simple`);
      return response.data;
    } catch (error) {
      console.error(`デモプロファイル '${profileName}' の取得に失敗:`, error);

      if (axios.isAxiosError(error)) {
        if (error.response?.status === 404) {
          throw new Error(`プロファイル '${profileName}' が見つかりません`);
        } else if (error.code === 'ECONNREFUSED') {
          throw new Error('バックエンドサーバーに接続できません。');
        } else if (error.response) {
          throw new Error(`HTTPエラー: ${error.response.status}`);
        }
      }

      throw new Error(`デモプロファイル '${profileName}' の取得に失敗しました`);
    }
  }

  /**
   * デモプロファイル統計情報を取得
   * 
   * @returns 統計情報（総数、タイプ別カウント）
   * @throws エラーメッセージ
   */
  async getDemoStats(): Promise<DemoStatsResponse> {
    try {
      const response = await this.api.get('/api/demo/stats');
      return response.data;
    } catch (error) {
      console.error('デモプロファイル統計情報の取得に失敗:', error);

      if (axios.isAxiosError(error)) {
        if (error.code === 'ECONNREFUSED') {
          throw new Error('バックエンドサーバーに接続できません。');
        } else if (error.response) {
          throw new Error(`HTTPエラー: ${error.response.status}`);
        }
      }

      throw new Error('デモプロファイル統計情報の取得に失敗しました');
    }
  }
}

// ========================================
// シングルトンインスタンスをエクスポート
// ========================================

export const apiService = new ApiService();
export default apiService;

// ========================================
// ユーティリティ関数
// ========================================

/**
 * API接続テスト
 * 
 * @returns サーバーが正常に動作している場合true
 */
export const testApiConnection = async (): Promise<boolean> => {
  return await apiService.healthCheck();
};