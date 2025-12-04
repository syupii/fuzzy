// frontend/src/services/api.ts - デモ機能なし版（27分野定義あり）

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// ==================== 型定義 ====================

export interface EvaluationRequest {
    // 基本項目
    research_intensity: number;
    advisor_style: number;
    team_work: number;
    workload: number;
    theory_practice: number;
    skill_development: number;
    lab_atmosphere: number;
    flexibility: number;
    publication_opportunity: number;
    interdisciplinary: number;
    communication_style: number;
    research_field_match: number;

    // 優先度
    research_intensity_priority?: number;
    advisor_style_priority?: number;
    team_work_priority?: number;
    workload_priority?: number;
    theory_practice_priority?: number;
    skill_development_priority?: number;
    lab_atmosphere_priority?: number;
    flexibility_priority?: number;
    publication_opportunity_priority?: number;
    interdisciplinary_priority?: number;
    communication_style_priority?: number;
    research_field_match_priority?: number;

    // 分野興味
    field_interests?: { [key: string]: number };
}

export type StudentProfile = EvaluationRequest;

export interface LabResult {
    lab_id?: string;
    lab_name: string;
    professor?: string;
    advisor?: string;
    professor_name?: string;
    research_area?: string;
    field_name?: string;
    field_id?: string;
    category?: string;

    overall_compatibility?: number;
    final_score?: number;
    basic_score?: number;
    field_score?: number;
    confidence?: number;
    priority_adjusted_score?: number;

    criteria_scores?: Record<string, number>;
    feature_scores?: Record<string, number>;

    recommendation?: string;
    explanation?: string;
    tree_path?: string;

    priority_analysis?: {
        high_priority_match: number;
        medium_priority_match: number;
        low_priority_match: number;
    };

    description?: string;
    specialization?: string;
    research_fields?: string[];
    features?: Record<string, number>;
    metadata?: any;
}

export interface EvaluationResponse {
    evaluation_results?: LabResult[];
    lab_results?: LabResult[];
    results?: LabResult[];
    student_profile?: any;
    summary?: any;
    metadata?: any;
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
    // テクノロジー・システム分野（10分野）
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

    // 教育・言語・文化分野（4分野）
    { id: 'japanese_education', name: '日本語教育・言語学', category: '教育・言語・文化' },
    { id: 'korean_studies', name: '韓国語・韓国文化研究', category: '教育・言語・文化' },
    { id: 'educational_tech', name: '教育工学・学習支援', category: '教育・言語・文化' },
    { id: 'english_humanities', name: '英語・人文学', category: '教育・言語・文化' },

    // 観光・地域分野（1分野）
    { id: 'tourism_regional', name: '観光情報・地域システム', category: '観光・地域' },

    // デザイン分野（4分野）
    { id: 'web_design_uiux', name: 'Webデザイン・UI/UX', category: 'デザイン' },
    { id: 'graphic_visual', name: 'グラフィック・視覚デザイン', category: 'デザイン' },
    { id: 'illustration_art', name: 'イラストレーション・アート', category: 'デザイン' },
    { id: 'design_thinking_marketing', name: 'デザイン思考・マーケティング', category: 'デザイン' },

    // 映像・音楽分野（4分野）
    { id: 'video_film', name: '映像制作・映画', category: '映像・音楽' },
    { id: 'animation', name: 'アニメーション', category: '映像・音楽' },
    { id: 'computer_music', name: 'コンピュータ音楽・サウンドアート', category: '映像・音楽' },
    { id: 'media_art', name: 'メディアアート', category: '映像・音楽' },

    // ゲーム・エンタメ分野（3分野）
    { id: 'game_dev', name: 'ゲーム開発', category: 'ゲーム・エンタメ' },
    { id: 'esports', name: 'eスポーツ', category: 'ゲーム・エンタメ' },
    { id: 'vr_ar_metaverse', name: 'VR/AR・メタバース', category: 'ゲーム・エンタメ' },

    // 人文・社会・体育分野（1分野）
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

// ==================== API関数 ====================

export const evaluateLabs = async (data: EvaluationRequest): Promise<EvaluationResponse> => {
    try {
        const response = await fetch(`${API_BASE_URL}/api/evaluate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }

        return await response.json();
    } catch (error: any) {
        console.error('評価API エラー:', error);
        throw new Error(error.message || '評価中にエラーが発生しました');
    }
};

export const evaluate = evaluateLabs;

export const getLabs = async () => {
    try {
        const response = await fetch(`${API_BASE_URL}/api/labs`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return await response.json();
    } catch (error: any) {
        console.error('研究室一覧取得エラー:', error);
        throw error;
    }
};

export const getSystemInfo = async () => {
    try {
        const response = await fetch(`${API_BASE_URL}/`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return await response.json();
    } catch (error: any) {
        console.error('システム情報取得エラー:', error);
        throw error;
    }
};

export const healthCheck = async () => {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return await response.json();
    } catch (error: any) {
        console.error('ヘルスチェックエラー:', error);
        throw error;
    }
};

// ==================== ヘルパー関数 ====================

export const getFieldName = (fieldId: string): string => {
    const field = RESEARCH_FIELDS.find(f => f.id === fieldId);
    return field ? field.name : fieldId;
};

export const getFieldCategory = (fieldId: string): string => {
    const field = RESEARCH_FIELDS.find(f => f.id === fieldId);
    return field ? field.category : '不明';
};

export const getFieldsByCategory = (category: string): ResearchField[] => {
    return RESEARCH_FIELDS.filter(f => f.category === category);
};

// ==================== apiService オブジェクト ====================

export const apiService = {
    getLabs,
    evaluateLabs,
    evaluate,
    getSystemInfo,
    healthCheck,
    getFieldName,
    getFieldCategory,
    getFieldsByCategory,
    RESEARCH_FIELDS,
    FIELD_CATEGORIES
};

export default apiService;