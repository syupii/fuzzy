// frontend/src/services/api.ts を修正

import axios from 'axios';

// バックエンドのURL設定（修正版）
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Axiosインスタンス作成
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000,
});

// レスポンスインターセプター（エラーハンドリング）
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

// 型定義（既存のものを維持）
export interface EvaluationPreferences {
  research_intensity: number;
  advisor_style: number;
  team_work: number;
  workload: number;
  theory_practice: number;
}

export interface Lab {
  id: string;
  name: string;
  advisor: string;
  description: string;
  research_intensity: number;
  advisor_style: number;
  team_work: number;
  workload: number;
  theory_practice: number;
  fields: string[];
  publications: number;
  funding: string;
  equipment: string;
  graduate_employment: string;
}

export interface EvaluationResult {
  lab_id: string;
  lab_name: string;
  advisor: string;
  overall_compatibility: number;
  feature_scores: {
    [key: string]: number;
  };
  confidence: number;
  recommendation: string;
  explanation: string;
}

export interface EvaluationResponse {
  student_profile: EvaluationPreferences;
  evaluation_results: EvaluationResult[];
  total_labs_evaluated: number;
  evaluation_timestamp: number;
  system_info: {
    fuzzy_enabled: boolean;
    genetic_enabled: boolean;
    evaluation_count: number;
  };
}

// API関数
export const apiService = {
  // ヘルスチェック
  async healthCheck() {
    try {
      const response = await api.get('/health');
      return response.data;
    } catch (error) {
      console.error('Health check failed:', error);
      throw error;
    }
  },

  // 研究室一覧取得
  async getLabs() {
    try {
      const response = await api.get('/api/labs');
      return response.data;
    } catch (error) {
      console.error('Failed to get labs:', error);
      throw error;
    }
  },

  // 適合度評価
  async evaluateCompatibility(preferences: EvaluationPreferences): Promise<EvaluationResponse> {
    try {
      const response = await api.post('/api/evaluate', preferences);
      return response.data;
    } catch (error) {
      console.error('Failed to evaluate compatibility:', error);
      throw error;
    }
  },

  // 最適化実行
  async optimize(studentProfiles: EvaluationPreferences[]) {
    try {
      const response = await api.post('/api/optimize', {
        student_profiles: studentProfiles
      });
      return response.data;
    } catch (error) {
      console.error('Failed to optimize:', error);
      throw error;
    }
  },

  // 説明取得
  async explainRecommendation(studentProfile: EvaluationPreferences, labId: string) {
    try {
      const response = await api.post('/api/explain', {
        student_profile: studentProfile,
        lab_id: labId
      });
      return response.data;
    } catch (error) {
      console.error('Failed to get explanation:', error);
      throw error;
    }
  },
};

export default api;