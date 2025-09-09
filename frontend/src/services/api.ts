// frontend/src/services/api.ts - 完全な型定義修正版

// 基本型定義
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
  innovation_risk?: number;
}

export interface Laboratory {
  id: string;
  name: string;
  professor: string;
  research_area?: string;
  specialization?: string;
  description?: string;
}

// 完全なLabResult型定義（すべての必要なプロパティを含む）
export interface LabResult {
  // 基本情報
  lab_id?: string;
  lab_name?: string;
  advisor?: string;
  professor?: string;

  // 研究分野情報
  research_area?: string;
  research_fields?: string[];
  specialization?: string;
  description?: string;

  // スコア関連
  overall_score?: number;
  compatibility_score?: number;
  field_match?: number;

  // 詳細評価
  compatibility?: {
    overall_score: number;
    criterion_scores?: { [key: string]: any };
  };

  // 推奨情報
  strengths?: string[];
  considerations?: string[];
  recommendations?: string[];

  // ランキング・メタデータ
  ranking_position?: number;
  metadata?: any;

  // 後方互換性のためのlab参照
  lab?: Laboratory;
}

export interface Faculty {
  name: string;
  specialties: string[];
}

export interface ResearchField {
  id: string;
  name: string;
  category: string;
  description?: string;
  faculty: Faculty[];
  faculty_count: number;
  keywords: string[];
}

// フィールド選択状態（チェックボックス用）
export interface FieldSelectionState {
  [fieldId: string]: boolean;
}

// 選択された分野の詳細評価
export interface SelectedFieldInterest {
  fieldId: string;
  interestLevel: number;    // 1-10
  experienceLevel: number;  // 1-10
  priority: number;         // 1から順位
}

export interface ResearchFieldInterests {
  [key: string]: number;
}

export interface StudentProfile {
  evaluation_criteria: EvaluationPreferences;
  field_interests: ResearchFieldInterests;
  selected_fields?: SelectedFieldInterest[];
  student_id?: string;
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

// 完全な研究分野定義（20分野）
export const RESEARCH_FIELDS: ResearchField[] = [
  // ===============================
  // テクノロジー・システム分野（12分野）
  // ===============================
  {
    id: 'ai_machine_learning',
    name: '人工知能・機械学習',
    category: 'テクノロジー・システム',
    description: 'データ解析、機械学習、深層学習、自然言語処理など',
    faculty: [
      { name: '伊藤雅彦', specialties: ['情報可視化', 'ユーザインタフェース', 'データ工学'] },
      { name: '内山敏雄', specialties: ['データ解析', '機械学習', 'レコメンド', 'テキストマイニング'] },
      { name: '小野亮太', specialties: ['人工知能', '情報工学', 'マルチエージェントシステム', '情報推薦'] },
      { name: '齋藤健司', specialties: ['人工知能', '教育システム', '仮想環境'] },
      { name: '谷口文武', specialties: ['機械学習', 'パターン認識'] },
      { name: '辻準平', specialties: ['社会シミュレーション', 'マルチエージェントシステム', 'IoT'] },
      { name: '山北貴典', specialties: ['データベース技術'] }
    ],
    faculty_count: 7,
    keywords: ['AI', '機械学習', 'データ分析', '深層学習']
  },
  {
    id: 'image_video_processing',
    name: '画像・映像処理',
    category: 'テクノロジー・システム',
    description: 'コンピュータビジョン、画像認識、医用画像工学など',
    faculty: [
      { name: '森圭佑', specialties: ['情報計測', '音声・画像情報処理', '医用情報処理', 'ゲームプログラミング'] },
      { name: '向田茂', specialties: ['画像処理', '顔学', '認知心理学', 'VR/AR', '3DCG'] },
      { name: '高井奈美', specialties: ['コンピュータグラフィックス', '画像処理', 'Webデザイン'] },
      { name: '藤原孝行', specialties: ['コンピュータビジョン', 'コンピュータグラフィックス'] },
      { name: '越野一博', specialties: ['医用画像工学', '数理統計学', '人工知能画像解析処理'] },
      { name: '上杉正人', specialties: ['医療情報システム開発', '医療言語処理', '画像処理'] }
    ],
    faculty_count: 6,
    keywords: ['画像処理', 'コンピュータビジョン', 'CG', '映像解析']
  },
  {
    id: 'network_security',
    name: 'ネットワーク・セキュリティ',
    category: 'テクノロジー・システム',
    description: 'ネットワーク管理、情報セキュリティ、通信システムなど',
    faculty: [
      { name: '尾崎宏和', specialties: ['コンピュータネットワーク', '通信システム', '信頼性'] },
      { name: '中島潤', specialties: ['情報通信ネットワーク', '情報セキュリティ', 'ITマネジメント'] },
      { name: '佐々木洋平', specialties: ['地球流体力学', '惑星科学', '計算機ネットワーク管理運用', 'セキュリティ'] }
    ],
    faculty_count: 3,
    keywords: ['ネットワーク', 'セキュリティ', '通信システム', '暗号化']
  },
  {
    id: 'database_information_systems',
    name: 'データベース・情報システム',
    category: 'テクノロジー・システム',
    description: 'データベース技術、経営情報システム、意思決定支援など',
    faculty: [
      { name: '山北貴典', specialties: ['データベース技術'] },
      { name: '坂田圭司', specialties: ['経営情報システム', '教育システム'] },
      { name: '向原強', specialties: ['オペレーションズ・リサーチ', '意思決定支援システム', '経営情報システム'] }
    ],
    faculty_count: 3,
    keywords: ['データベース', '情報システム', 'SQL', 'データ管理']
  },
  {
    id: 'embedded_iot',
    name: '組込み・IoT',
    category: 'テクノロジー・システム',
    description: '組込みシステム、IoT、ユビキタスコンピューティングなど',
    faculty: [
      { name: '田鎖次郎', specialties: ['組込みシステム工学', '情報倫理'] },
      { name: '湯村翼', specialties: ['ユビキタスコンピューティング', 'ヒューマンコンピュータインタラクション', '地球惑星科学'] }
    ],
    faculty_count: 2,
    keywords: ['組込み', 'IoT', 'センサー', 'マイコン']
  },
  {
    id: 'education_linguistics',
    name: '教育・言語学',
    category: 'テクノロジー・システム',
    description: '日本語教育、多言語教育、教育システム、語学教育など',
    faculty: [
      { name: '飯嶋美知子', specialties: ['日本語教育学', '日中対照言語学'] },
      { name: '金銀珠', specialties: ['日韓対照言語学', '日本語教育', '韓国語教育', '複言語教育'] },
      { name: '田中英夫', specialties: ['国際経営論', '国際関係論', '中国語教育'] },
      { name: '齋藤一', specialties: ['観光情報学', '教育工学'] },
      { name: '近澤潤', specialties: ['発想法', 'デザイン思考', 'イノベーション教育'] }
    ],
    faculty_count: 5,
    keywords: ['日本語教育', '多言語', '教育システム', '語学']
  },
  {
    id: 'natural_science_mathematics',
    name: '自然科学・数理',
    category: 'テクノロジー・システム',
    description: '宇宙科学、地球科学、統計解析、数値計算、気象現象など',
    faculty: [
      { name: '柿並義宏', specialties: ['宇宙科学', '地球惑星科学', '大気科学', '動物行動学'] },
      { name: '甫喜本司', specialties: ['データ解析法', '統計数理', '時間的・空間的な現象の予測方法'] },
      { name: '松井伸也', specialties: ['非線形現象の解析', '流体現象', '気象現象', '反応拡散系'] },
      { name: '新井山亮', specialties: ['社会情報工学', '光・波動電子工学', '数値解析'] },
      { name: '佐々木洋平', specialties: ['地球流体力学', '惑星科学', '応用数学', '数値計算'] },
      { name: '湯村翼', specialties: ['地球惑星科学', 'ヒューマンコンピュータインタラクション'] }
    ],
    faculty_count: 6,
    keywords: ['宇宙科学', '地球科学', '統計解析', '数値計算']
  },
  {
    id: 'medical_informatics',
    name: '医療情報・ヘルスケア',
    category: 'テクノロジー・システム',
    description: '医用画像工学、医療情報システム、医療データ解析など',
    faculty: [
      { name: '越野一博', specialties: ['医用画像工学', '数理統計学', '人工知能画像解析処理'] },
      { name: '上杉正人', specialties: ['医療情報システム開発', '医療言語処理', '画像処理'] }
    ],
    faculty_count: 2,
    keywords: ['医療IT', '医用画像', 'ヘルスケア', '医療データ']
  },
  {
    id: 'tourism_regional_systems',
    name: '観光情報・地域システム',
    category: 'テクノロジー・システム',
    description: '観光情報学、地域情報システム、観光データ分析など',
    faculty: [
      { name: '齋藤一', specialties: ['観光情報学', '教育工学'] },
      { name: '小野亮太', specialties: ['人工知能', '情報工学', '観光情報'] }
    ],
    faculty_count: 2,
    keywords: ['観光情報', '地域システム', '観光データ', '地域活性化']
  },
  {
    id: 'business_information_systems',
    name: '経営情報・意思決定支援',
    category: 'テクノロジー・システム',
    description: '経営情報システム、意思決定支援、経営データ分析など',
    faculty: [
      { name: '坂田圭司', specialties: ['経営情報システム', '教育システム'] },
      { name: '向原強', specialties: ['オペレーションズ・リサーチ', '意思決定支援システム', '経営情報システム'] },
      { name: '田中英夫', specialties: ['国際経営論', '国際関係論'] }
    ],
    faculty_count: 3,
    keywords: ['経営情報', '意思決定', 'OR', '経営データ分析']
  },
  {
    id: 'audio_sound_processing',
    name: '音声・音響情報処理',
    category: 'テクノロジー・システム',
    description: '音声情報処理、音響技術、システム運用など',
    faculty: [
      { name: '廣奥透', specialties: ['音声情報処理', 'ソフトウェア開発', 'コンピュータシステム運用'] },
      { name: '森圭佑', specialties: ['情報計測', '音声・画像情報処理', '医用情報処理'] }
    ],
    faculty_count: 2,
    keywords: ['音声処理', '音響技術', 'システム運用', '音声認識']
  },

  // ===============================
  // クリエイティブ分野（4分野）
  // ===============================
  {
    id: 'web_design_ui_ux',
    name: 'Webデザイン・UI/UX',
    category: 'クリエイティブ',
    description: 'Webデザイン、UX/UIデザイン、インタフェースデザインなど',
    faculty: [
      { name: '杉沢愛美', specialties: ['Webデザイン', 'グラフィックデザイン', 'UX・UIデザイン', 'ブランディングデザイン'] },
      { name: '坂本牧葉', specialties: ['視覚デザイン', 'インタフェースデザイン', '感性工学', 'イラストレーション'] },
      { name: '高井奈美', specialties: ['コンピュータグラフィックス', '画像処理', 'Webデザイン'] },
      { name: '安田光孝', specialties: ['UX/UIデザイン', 'コンテンツプロデュース', 'アントレプレナーシップ教育'] }
    ],
    faculty_count: 4,
    keywords: ['Webデザイン', 'UX/UI', 'ブランディング', 'インタフェース']
  },
  {
    id: 'design_visual_expression',
    name: 'デザイン・視覚表現',
    category: 'クリエイティブ',
    description: '視覚デザイン、グラフィックデザイン、感性工学など',
    faculty: [
      { name: '坂本牧葉', specialties: ['視覚デザイン', 'インタフェースデザイン', '感性工学', 'イラストレーション'] },
      { name: '大嶋宏一', specialties: ['映像表現', 'アニメーション表現', 'メディア表現', '視覚芸術'] },
      { name: 'Marty M. ITO', specialties: ['イラストレーション', 'ローブロアート', 'アートマネジメント'] },
      { name: '安田光孝', specialties: ['UX/UIデザイン', 'コンテンツプロデュース', 'デザイン思考'] }
    ],
    faculty_count: 4,
    keywords: ['視覚デザイン', 'グラフィック', 'イラスト', 'アート']
  },
  {
    id: 'video_animation',
    name: '映像・アニメーション',
    category: 'クリエイティブ',
    description: '映像制作、アニメーション表現、メディア表現など',
    faculty: [
      { name: '大嶋宏一', specialties: ['映像表現', 'アニメーション表現', 'メディア表現', '視覚芸術'] },
      { name: '島田映二', specialties: ['映像制作', '映像表現'] }
    ],
    faculty_count: 2,
    keywords: ['映像制作', 'アニメーション', 'メディア表現', '動画編集']
  },
  {
    id: 'computer_music_sound_art',
    name: 'コンピュータ音楽・サウンドアート',
    category: 'クリエイティブ',
    description: 'コンピュータ音楽、サウンドアート、音響技術など',
    faculty: [
      { name: '平山遙香', specialties: ['コンピュータ音楽', 'サウンドアート', '現代音楽'] },
      { name: '廣奥透', specialties: ['音声情報処理', 'ソフトウェア開発', 'コンピュータシステム運用'] }
    ],
    faculty_count: 2,
    keywords: ['コンピュータ音楽', 'サウンドアート', 'DTM', '音響技術']
  },

  // ===============================
  // エンターテイメント分野（2分野）
  // ===============================
  {
    id: 'game_development_esports',
    name: 'ゲーム開発・eスポーツ',
    category: 'エンターテイメント',
    description: 'ゲームプログラミング、eスポーツ、メタバースなど',
    faculty: [
      { name: '森川悟', specialties: ['ゲームプログラミング'] },
      { name: '川原勝', specialties: ['eスポーツ', 'メタバース', '教育学'] }
    ],
    faculty_count: 2,
    keywords: ['ゲーム開発', 'eスポーツ', 'メタバース', 'Unity']
  },
  {
    id: 'vr_ar_media_art',
    name: 'VR/AR・メディアアート',
    category: 'エンターテイメント',
    description: 'VR/AR技術、3DCG、メディアアート、認知心理学など',
    faculty: [
      { name: '向田茂', specialties: ['画像処理', '顔学', '認知心理学', 'VR/AR', '3DCG', 'メディアアート'] },
      { name: '隼田尚彦', specialties: ['環境行動学', '地域コミュニティ', 'インターフェイス', 'メディアアート'] }
    ],
    faculty_count: 2,
    keywords: ['VR', 'AR', 'メディアアート', '3DCG']
  },

  // ===============================
  // 人文・社会・体育分野（2分野）
  // ===============================
  {
    id: 'philosophy_humanities',
    name: '哲学・人文・環境行動学',
    category: '人文・社会・体育',
    description: '哲学、倫理学、芸術学、環境行動学、地域コミュニティなど',
    faculty: [
      { name: '三浦洋', specialties: ['哲学', '倫理学', '芸術学'] },
      { name: '隼田尚彦', specialties: ['環境行動学', '地域コミュニティ', '建築計画学'] }
    ],
    faculty_count: 2,
    keywords: ['哲学', '倫理学', '環境行動', '地域研究']
  },
  {
    id: 'sports_exercise_science',
    name: 'スポーツ・体育科学',
    category: '人文・社会・体育',
    description: 'スポーツバイオメカニクス、トレーニング科学、体育実践など',
    faculty: [
      { name: '綿谷貴志', specialties: ['スポーツバイオメカニクス', 'トレーニング科学'] },
      { name: '織田哲', specialties: ['体育'] }
    ],
    faculty_count: 2,
    keywords: ['スポーツ科学', 'バイオメカニクス', '体育', '運動解析']
  }
];

// 分野カテゴリ（拡張版）
export const FIELD_CATEGORIES = [
  'テクノロジー・システム',
  'クリエイティブ',
  'エンターテイメント',
  '人文・社会・体育'
];

// 評価基準情報（13項目・innovation_risk追加）
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

// フィールドユーティリティ関数
export const fieldUtils = {
  // カテゴリ別の分野取得
  getFieldsByCategory: (category: string): ResearchField[] => {
    return RESEARCH_FIELDS.filter(field => field.category === category);
  },

  // 分野IDから詳細情報取得
  getFieldById: (fieldId: string): ResearchField | undefined => {
    return RESEARCH_FIELDS.find(field => field.id === fieldId);
  },

  // 教員名をカンマ区切りで取得
  getFacultyNames: (fieldId: string): string => {
    const field = fieldUtils.getFieldById(fieldId);
    return field ? field.faculty.map(f => f.name).join('、') : '';
  },

  // キーワード検索
  searchByKeyword: (keyword: string): ResearchField[] => {
    const lowerKeyword = keyword.toLowerCase();
    return RESEARCH_FIELDS.filter(field =>
      field.keywords.some(k => k.toLowerCase().includes(lowerKeyword)) ||
      field.name.toLowerCase().includes(lowerKeyword) ||
      field.description?.toLowerCase().includes(lowerKeyword)
    );
  },

  // 教員名検索
  searchByFaculty: (facultyName: string): ResearchField[] => {
    return RESEARCH_FIELDS.filter(field =>
      field.faculty.some(f => f.name.includes(facultyName))
    );
  }
};

// APIサービス
class ApiService {
  private baseURL = 'http://localhost:8000';

  async evaluateLabs(preferences: EvaluationPreferences, selectedFields?: SelectedFieldInterest[]): Promise<EvaluationResponse> {
    try {
      const requestData = {
        evaluation_criteria: preferences,
        selected_fields: selectedFields || []
      };

      const response = await fetch(`${this.baseURL}/api/evaluate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData)
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      // レスポンスデータを正規化
      const normalizedResults: LabResult[] = (data.lab_results || data.results || []).map((item: any, index: number) => ({
        lab_id: item.lab_id || item.id,
        lab_name: item.lab_name || item.name || `研究室${index + 1}`,
        advisor: item.advisor || item.professor,
        overall_score: item.compatibility_score || item.overall_score || 0,
        compatibility_score: item.compatibility_score || item.overall_score || 0,
        field_match: item.field_match || 0,
        research_fields: item.research_fields || [],
        specialization: item.specialization || '',
        description: item.description || '詳細情報はありません',
        strengths: item.strengths || [],
        considerations: item.considerations || [],
        recommendations: item.recommendations || [],
        compatibility: item.compatibility || {
          overall_score: item.compatibility_score || item.overall_score || 0,
          criterion_scores: item.criterion_scores || {}
        },
        metadata: item.metadata || {},
        // 後方互換性
        lab: {
          id: item.lab_id || item.id || `lab_${index}`,
          name: item.lab_name || item.name || `研究室${index + 1}`,
          professor: item.advisor || item.professor || '指導教員情報なし',
          research_area: item.research_area || '',
          specialization: item.specialization || '',
          description: item.description || ''
        }
      }));

      return {
        lab_results: normalizedResults,
        results: normalizedResults,
        summary: {
          total_labs: data.total_labs || normalizedResults.length,
          avg_score: data.avg_score || (normalizedResults.length > 0 ?
            normalizedResults.reduce((sum, lab) => sum + (lab.overall_score || 0), 0) / normalizedResults.length : 0),
          best_match_lab: normalizedResults.length > 0 ? normalizedResults[0].lab_name : undefined,
          recommendations: data.recommendations || []
        },
        metadata: data.metadata || {}
      };

    } catch (error) {
      console.error('Lab evaluation failed:', error);
      throw error;
    }
  }

  async getDemoProfile(): Promise<StudentProfile> {
    return {
      evaluation_criteria: {
        research_intensity: 7,
        advisor_style: 6,
        team_work: 7,
        workload: 6,
        theory_practice: 7,
        research_field_match: 8,
        skill_development: 7,
        lab_atmosphere: 8,
        flexibility: 7,
        publication_opportunity: 6,
        interdisciplinary: 6,
        communication_style: 7,
        innovation_risk: 7
      },
      field_interests: {
        'ai_machine_learning': 9,
        'web_design_ui_ux': 7,
        'game_development_esports': 6
      }
    };
  }
}

export const apiService = new ApiService();