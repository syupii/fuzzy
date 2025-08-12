import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

// Axiosインスタンス作成
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 10000, // 10秒タイムアウト
});

// 型定義
export interface Lab {
  id: number;
  name: string;
  professor: string;
  research_area: string;
  description: string;
  features: {
    research_intensity: number;
    advisor_style: number;
    team_work: number;
    workload: number;
    theory_practice: number;
  };
  created_at: string;
}

export interface EvaluationPreferences {
  research_intensity: number;
  advisor_style: number;
  team_work: number;
  workload: number;
  theory_practice: number;
}

export interface CompatibilityResult {
  overall_score: number;
  criterion_scores: {
    [key: string]: {
      similarity: number;
      weighted_score: number;
      user_preference: number;
      lab_feature: number;
      weight: number;
    };
  };
  confidence: number;
  weights_used: number[];
  explanation: string;
}

export interface EvaluationResult {
  lab: Lab;
  compatibility: CompatibilityResult;
}

export interface EvaluationSummary {
  total_labs: number;
  best_match: string;
  avg_score: number;
  evaluation_id: number;
  session_id: string;
}

export interface EvaluationResponse {
  results: EvaluationResult[];
  summary: EvaluationSummary;
  algorithm_info: {
    engine: string;
    criteria_weights: { [key: string]: number };
  };
}

// API関数
export const apiService = {
  // ヘルスチェック
  async healthCheck() {
    const response = await api.get('/health');
    return response.data;
  },

  // 適合度評価
  async evaluateCompatibility(preferences: EvaluationPreferences): Promise<EvaluationResponse> {
    const response = await api.post('/evaluate', preferences);
    return response.data;
  },

  // デモデータ取得
  async getDemoData(): Promise<{ demo_preferences: EvaluationPreferences; message: string }> {
    const response = await api.get('/demo-data');
    return response.data;
  },
};

export default api;