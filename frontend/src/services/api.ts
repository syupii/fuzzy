import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

// Axiosインスタンス作成
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 10000,
});

// 研究分野の型定義
export interface ResearchField {
  id: string;
  name: string;
  description: string;
  category: string;
  keywords: string[];
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  marketDemand: 'high' | 'medium' | 'low';
  academicLevel: 'undergraduate' | 'graduate' | 'both';
}

// プログラミング言語の型定義
export interface ProgrammingLanguage {
  id: string;
  name: string;
  category: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  applications: string[];
  frameworks: string[];
  marketDemand: 'high' | 'medium' | 'low';
  learningCurve: number; // 1-10
}

// 技術フレームワーク・ツールの型定義
export interface TechFramework {
  id: string;
  name: string;
  category: string;
  language?: string;
  type?: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  popularity: number; // 1-10
  learningResources: 'abundant' | 'moderate' | 'limited';
}

export interface FieldInterest {
  isSelected: boolean;
  interestLevel: number;
  priority: 'high' | 'medium' | 'low';
}

export interface TechStackPreference {
  languagePreferences: string[];
  frameworkExperience: string[];
  learningWillingness: number;
  careerGoals: string[];
}

// 既存の型定義（互換性維持）
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
    research_field_match: number;
    skill_development: number;
    learning_pace: number;
    difficulty_preference: number;
    communication_style: number;
    meeting_frequency: number;
    lab_atmosphere: number;
    innovation_risk: number;
    methodology_preference: number;
    interdisciplinary: number;
    flexibility: number;
    evening_weekend_work: number;
    publication_opportunity: number;
    financial_support: number;
    lab_hierarchy: number;
    core_time_flexibility: number;
  };
  created_at: string;
}

export interface EvaluationPreferences {
  research_intensity: number;
  advisor_style: number;
  team_work: number;
  workload: number;
  theory_practice: number;
  research_field_match: number;
  skill_development: number;
  learning_pace: number;
  difficulty_preference: number;
  communication_style: number;
  meeting_frequency: number;
  lab_atmosphere: number;
  innovation_risk: number;
  methodology_preference: number;
  interdisciplinary: number;
  flexibility: number;
  evening_weekend_work: number;
  publication_opportunity: number;
  financial_support: number;
  lab_hierarchy: number;
  core_time_flexibility: number;
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
  field_matching?: {
    matched_fields: string[];
    field_scores: { [fieldId: string]: number };
    field_weight: number;
  };
}

export interface EvaluationResponse {
  results: {
    lab: Lab;
    compatibility: CompatibilityResult;
  }[];
  summary: {
    total_labs: number;
    avg_score: number;
    max_score: number;
    field_analysis?: {
      selected_fields_count: number;
      average_interest: number;
      primary_category: string;
      field_coverage: number;
    };
  };
  session_id: string;
}

// 拡張された研究分野データ（30項目）
export const RESEARCH_FIELDS: ResearchField[] = [
  // 既存項目（改良版）
  {
    id: 'ai',
    name: '人工知能・機械学習',
    description: 'AI、機械学習、深層学習の研究と応用',
    category: '情報工学・AI',
    keywords: ['AI', '機械学習', 'ディープラーニング', 'ニューラルネット'],
    difficulty: 'intermediate',
    marketDemand: 'high',
    academicLevel: 'both'
  },
  {
    id: 'cv',
    name: 'コンピュータビジョン',
    description: '画像認識、映像解析、パターン認識技術',
    category: '情報工学・AI',
    keywords: ['画像処理', '映像認識', 'パターン認識', 'OpenCV'],
    difficulty: 'intermediate',
    marketDemand: 'high',
    academicLevel: 'both'
  },
  {
    id: 'nlp',
    name: '自然言語処理',
    description: 'テキスト解析、言語理解、対話システム',
    category: '情報工学・AI',
    keywords: ['テキストマイニング', '言語モデル', '翻訳', 'ChatGPT'],
    difficulty: 'advanced',
    marketDemand: 'high',
    academicLevel: 'both'
  },
  {
    id: 'data_science',
    name: 'データサイエンス・ビッグデータ',
    description: 'ビッグデータ解析、統計分析、予測モデリング',
    category: 'データ科学',
    keywords: ['ビッグデータ', '統計', '予測', 'Python', 'R'],
    difficulty: 'intermediate',
    marketDemand: 'high',
    academicLevel: 'both'
  },

  // Web・モバイル開発分野
  {
    id: 'web_development',
    name: 'Webアプリケーション開発',
    description: 'フロントエンド・バックエンド開発、SPA構築',
    category: 'Web・アプリ開発',
    keywords: ['React', 'Vue', 'Node.js', 'API', 'レスポンシブ'],
    difficulty: 'beginner',
    marketDemand: 'high',
    academicLevel: 'undergraduate'
  },
  {
    id: 'mobile_development',
    name: 'モバイルアプリケーション',
    description: 'iOS・Android開発、クロスプラットフォーム',
    category: 'Web・アプリ開発',
    keywords: ['iOS', 'Android', 'Flutter', 'React Native', 'Swift'],
    difficulty: 'intermediate',
    marketDemand: 'high',
    academicLevel: 'undergraduate'
  },
  {
    id: 'game_development',
    name: 'ゲーム開発・インタラクティブメディア',
    description: 'ゲームエンジン、3Dグラフィックス、VR/AR',
    category: 'エンターテイメント',
    keywords: ['Unity', '3D', 'VR', 'AR', 'ゲームエンジン'],
    difficulty: 'intermediate',
    marketDemand: 'medium',
    academicLevel: 'both'
  },

  // 基盤技術・システム分野
  {
    id: 'software_engineering',
    name: 'ソフトウェア工学・システム設計',
    description: 'ソフトウェア設計、開発手法、品質管理、DevOps',
    category: '基盤技術',
    keywords: ['設計パターン', 'アジャイル', 'DevOps', 'テスト'],
    difficulty: 'intermediate',
    marketDemand: 'high',
    academicLevel: 'both'
  },
  {
    id: 'database',
    name: 'データベース・データ管理',
    description: 'DB設計、ビッグデータ、データウェアハウス',
    category: '基盤技術',
    keywords: ['SQL', 'NoSQL', 'MongoDB', 'PostgreSQL', 'データモデリング'],
    difficulty: 'intermediate',
    marketDemand: 'high',
    academicLevel: 'both'
  },
  {
    id: 'network',
    name: 'ネットワーク・インフラ技術',
    description: 'ネットワーク設計、クラウド、分散システム',
    category: '基盤技術',
    keywords: ['TCP/IP', 'クラウド', 'AWS', 'ネットワーク設計'],
    difficulty: 'intermediate',
    marketDemand: 'high',
    academicLevel: 'both'
  },
  {
    id: 'cybersecurity',
    name: 'サイバーセキュリティ・情報セキュリティ',
    description: '情報セキュリティ、暗号化、セキュリティ監査',
    category: '基盤技術',
    keywords: ['暗号', 'ペネトレーションテスト', 'セキュリティ', 'CISSP'],
    difficulty: 'advanced',
    marketDemand: 'high',
    academicLevel: 'both'
  },

  // 北海道情報大学特化分野
  {
    id: 'medical_informatics',
    name: '医療情報学・ヘルスケアIT',
    description: '電子カルテ、医用画像処理、遠隔医療システム',
    category: '医療・健康',
    keywords: ['電子カルテ', '医用画像', '遠隔医療', 'HL7', 'DICOM'],
    difficulty: 'advanced',
    marketDemand: 'high',
    academicLevel: 'both'
  },
  {
    id: 'business_informatics',
    name: '経営情報学・ビジネスIT',
    description: 'ERP、CRM、BI、デジタル変革（DX）',
    category: 'ビジネス・経営',
    keywords: ['ERP', 'CRM', 'BI', 'DX', 'データ分析'],
    difficulty: 'intermediate',
    marketDemand: 'high',
    academicLevel: 'both'
  },
  {
    id: 'media_processing',
    name: 'デジタルメディア・コンテンツ制作',
    description: '映像処理、音響処理、CG・アニメーション',
    category: 'メディア・コンテンツ',
    keywords: ['映像編集', '音響', 'CG', 'After Effects', 'Blender'],
    difficulty: 'intermediate',
    marketDemand: 'medium',
    academicLevel: 'undergraduate'
  },
  {
    id: 'hci_ux',
    name: 'HCI・UI/UXデザイン',
    description: 'ユーザビリティ、インタラクションデザイン、アクセシビリティ',
    category: 'デザイン・UX',
    keywords: ['ユーザビリティ', 'Figma', 'プロトタイピング', 'アクセシビリティ'],
    difficulty: 'intermediate',
    marketDemand: 'high',
    academicLevel: 'undergraduate'
  },

  // IoT・組み込み・ロボティクス
  {
    id: 'iot_embedded',
    name: 'IoT・組み込みシステム',
    description: 'センサーネットワーク、組み込みプログラミング、リアルタイムOS',
    category: '組み込み・IoT',
    keywords: ['Arduino', 'Raspberry Pi', 'センサー', 'マイコン', 'RTOS'],
    difficulty: 'advanced',
    marketDemand: 'medium',
    academicLevel: 'both'
  },
  {
    id: 'robotics',
    name: 'ロボティクス・制御工学',
    description: 'ロボット工学、制御システム、自律移動システム',
    category: '組み込み・IoT',
    keywords: ['ROS', '制御理論', '自律移動', 'センサー融合'],
    difficulty: 'advanced',
    marketDemand: 'medium',
    academicLevel: 'both'
  },

  // 新興・先端技術
  {
    id: 'blockchain',
    name: 'ブロックチェーン・分散台帳技術',
    description: '暗号通貨、スマートコントラクト、DeFi、NFT',
    category: '新興技術',
    keywords: ['Bitcoin', 'Ethereum', 'スマートコントラクト', 'DeFi', 'NFT'],
    difficulty: 'advanced',
    marketDemand: 'medium',
    academicLevel: 'graduate'
  },
  {
    id: 'cloud_computing',
    name: 'クラウドコンピューティング',
    description: 'AWS、Azure、マイクロサービス、サーバーレス',
    category: '基盤技術',
    keywords: ['AWS', 'Azure', 'マイクロサービス', 'Docker', 'Kubernetes'],
    difficulty: 'intermediate',
    marketDemand: 'high',
    academicLevel: 'both'
  },
  {
    id: 'quantum',
    name: '量子コンピューティング',
    description: '量子アルゴリズム、量子情報理論、量子機械学習',
    category: '先端技術',
    keywords: ['Qiskit', '量子ビット', '量子もつれ', '量子アルゴリズム'],
    difficulty: 'advanced',
    marketDemand: 'low',
    academicLevel: 'graduate'
  },

  // 理論・数学・アルゴリズム
  {
    id: 'algorithms',
    name: 'アルゴリズム・計算量理論',
    description: '最適化、グラフ理論、並列アルゴリズム',
    category: '理論・数学',
    keywords: ['グラフ理論', '最適化', 'NP問題', '並列処理'],
    difficulty: 'advanced',
    marketDemand: 'medium',
    academicLevel: 'graduate'
  },
  {
    id: 'optimization',
    name: '最適化・オペレーションズリサーチ',
    description: '線形計画法、遺伝的アルゴリズム、メタヒューリスティック',
    category: '理論・数学',
    keywords: ['線形計画', '遺伝的アルゴリズム', 'シミュレーション', 'OR'],
    difficulty: 'advanced',
    marketDemand: 'medium',
    academicLevel: 'graduate'
  },

  // 学際・応用分野
  {
    id: 'bioinformatics',
    name: 'バイオインフォマティクス',
    description: 'ゲノム解析、生命情報学、バイオデータ解析',
    category: '生命科学',
    keywords: ['ゲノム', 'DNA', 'タンパク質', 'バイオデータ'],
    difficulty: 'advanced',
    marketDemand: 'low',
    academicLevel: 'graduate'
  },
  {
    id: 'environmental_informatics',
    name: '環境情報学・地球科学情報',
    description: '気象データ解析、環境モニタリング、GIS',
    category: '環境・地球科学',
    keywords: ['GIS', '気象データ', '環境センサー', 'リモートセンシング'],
    difficulty: 'intermediate',
    marketDemand: 'medium',
    academicLevel: 'both'
  },
  {
    id: 'regional_informatics',
    name: '地域情報システム・社会応用',
    description: '観光情報、農業ICT、防災システム、スマートシティ',
    category: '社会応用',
    keywords: ['観光', '農業', '防災', 'スマートシティ', '地方創生'],
    difficulty: 'intermediate',
    marketDemand: 'medium',
    academicLevel: 'undergraduate'
  },

  // 形式手法・検証技術
  {
    id: 'formal_methods',
    name: '形式手法・プログラム検証',
    description: 'モデル検査、定理証明、仕様記述言語',
    category: '理論・検証',
    keywords: ['モデル検査', '定理証明', 'TLA+', 'Coq', '形式仕様'],
    difficulty: 'advanced',
    marketDemand: 'low',
    academicLevel: 'graduate'
  }
];

// プログラミング言語データ（15項目）
export const PROGRAMMING_LANGUAGES: ProgrammingLanguage[] = [
  // メイン言語（基礎・汎用）
  {
    id: 'python',
    name: 'Python',
    category: '汎用・スクリプト',
    difficulty: 'beginner',
    applications: ['AI/ML', 'データ分析', 'Web開発', '自動化', '科学計算'],
    frameworks: ['Django', 'Flask', 'FastAPI', 'pandas', 'scikit-learn', 'TensorFlow', 'PyTorch'],
    marketDemand: 'high',
    learningCurve: 3
  },
  {
    id: 'java',
    name: 'Java',
    category: '汎用・エンタープライズ',
    difficulty: 'intermediate',
    applications: ['エンタープライズ開発', 'Android開発', '大規模システム', 'バックエンド'],
    frameworks: ['Spring', 'Spring Boot', 'Maven', 'Gradle', 'Hibernate'],
    marketDemand: 'high',
    learningCurve: 5
  },
  {
    id: 'javascript',
    name: 'JavaScript',
    category: 'Web・フロントエンド',
    difficulty: 'beginner',
    applications: ['フロントエンド開発', 'バックエンド（Node.js）', 'SPA', 'モバイルアプリ'],
    frameworks: ['React', 'Vue.js', 'Angular', 'Node.js', 'Express', 'Next.js'],
    marketDemand: 'high',
    learningCurve: 4
  },
  {
    id: 'typescript',
    name: 'TypeScript',
    category: 'Web・フロントエンド',
    difficulty: 'intermediate',
    applications: ['型安全Web開発', 'Angular開発', '大規模JavaScript', 'React開発'],
    frameworks: ['Angular', 'Next.js', 'Nest.js', 'Vue 3'],
    marketDemand: 'high',
    learningCurve: 5
  },
  {
    id: 'html_css',
    name: 'HTML/CSS',
    category: 'Web・マークアップ',
    difficulty: 'beginner',
    applications: ['Webページ構築', 'レスポンシブデザイン', 'UI実装'],
    frameworks: ['Bootstrap', 'Tailwind CSS', 'Sass', 'SCSS'],
    marketDemand: 'high',
    learningCurve: 2
  },
  {
    id: 'swift',
    name: 'Swift',
    category: 'モバイル・iOS',
    difficulty: 'intermediate',
    applications: ['iOS開発', 'macOS開発', 'watchOS開発', 'tvOS開発'],
    frameworks: ['UIKit', 'SwiftUI', 'Core Data', 'Combine'],
    marketDemand: 'high',
    learningCurve: 5
  },
  {
    id: 'kotlin',
    name: 'Kotlin',
    category: 'モバイル・Android',
    difficulty: 'intermediate',
    applications: ['Android開発', 'マルチプラットフォーム', 'サーバーサイド'],
    frameworks: ['Android Jetpack', 'Ktor', 'Kotlin Multiplatform'],
    marketDemand: 'high',
    learningCurve: 5
  },
  {
    id: 'cpp',
    name: 'C++',
    category: 'システム・高性能',
    difficulty: 'advanced',
    applications: ['システム開発', 'ゲーム開発', '高性能計算', '組み込み'],
    frameworks: ['Qt', 'Boost', 'OpenCV', 'Unreal Engine'],
    marketDemand: 'medium',
    learningCurve: 8
  },
  {
    id: 'csharp',
    name: 'C#',
    category: 'エンタープライズ・Microsoft',
    difficulty: 'intermediate',
    applications: ['Windows開発', 'Web開発', 'Unity開発', 'クラウド開発'],
    frameworks: ['.NET', 'ASP.NET Core', 'Unity', 'Xamarin', 'Blazor'],
    marketDemand: 'high',
    learningCurve: 5
  },
  {
    id: 'php',
    name: 'PHP',
    category: 'Web・サーバーサイド',
    difficulty: 'beginner',
    applications: ['サーバーサイド開発', 'WordPress開発', 'CMS構築', 'Web API'],
    frameworks: ['Laravel', 'Symfony', 'CodeIgniter', 'WordPress'],
    marketDemand: 'medium',
    learningCurve: 3
  },
  {
    id: 'r',
    name: 'R',
    category: 'データ分析・統計',
    difficulty: 'intermediate',
    applications: ['統計解析', 'データ可視化', '学術研究', 'バイオインフォマティクス'],
    frameworks: ['ggplot2', 'dplyr', 'shiny', 'tidyverse'],
    marketDemand: 'medium',
    learningCurve: 6
  },
  {
    id: 'sql',
    name: 'SQL',
    category: 'データベース',
    difficulty: 'beginner',
    applications: ['データベース操作', 'データ分析', 'レポート作成', 'ETL'],
    frameworks: ['MySQL', 'PostgreSQL', 'SQLite', 'Microsoft SQL Server'],
    marketDemand: 'high',
    learningCurve: 3
  },
  {
    id: 'go',
    name: 'Go',
    category: 'システム・バックエンド',
    difficulty: 'intermediate',
    applications: ['マイクロサービス', 'API開発', '高性能Web', 'DevOps'],
    frameworks: ['Gin', 'Echo', 'Beego', 'Buffalo'],
    marketDemand: 'medium',
    learningCurve: 4
  },
  {
    id: 'dart',
    name: 'Dart',
    category: 'モバイル・クロスプラットフォーム',
    difficulty: 'intermediate',
    applications: ['Flutter開発', 'クロスプラットフォーム', 'Web開発'],
    frameworks: ['Flutter', 'AngularDart'],
    marketDemand: 'medium',
    learningCurve: 4
  },
  {
    id: 'rust',
    name: 'Rust',
    category: 'システム・安全性',
    difficulty: 'advanced',
    applications: ['システムプログラミング', 'WebAssembly', 'ブロックチェーン'],
    frameworks: ['Actix', 'Rocket', 'Tokio'],
    marketDemand: 'medium',
    learningCurve: 8
  }
];

// 技術フレームワーク・ツールデータ（20項目）
export const TECH_FRAMEWORKS: TechFramework[] = [
  // Web開発フレームワーク
  {
    id: 'react',
    name: 'React.js',
    category: 'フロントエンド',
    language: 'JavaScript',
    difficulty: 'intermediate',
    popularity: 9,
    learningResources: 'abundant'
  },
  {
    id: 'vue',
    name: 'Vue.js',
    category: 'フロントエンド',
    language: 'JavaScript',
    difficulty: 'beginner',
    popularity: 8,
    learningResources: 'abundant'
  },
  {
    id: 'angular',
    name: 'Angular',
    category: 'フロントエンド',
    language: 'TypeScript',
    difficulty: 'advanced',
    popularity: 7,
    learningResources: 'abundant'
  },
  {
    id: 'nodejs',
    name: 'Node.js',
    category: 'バックエンド',
    language: 'JavaScript',
    difficulty: 'intermediate',
    popularity: 9,
    learningResources: 'abundant'
  },
  {
    id: 'django',
    name: 'Django',
    category: 'バックエンド',
    language: 'Python',
    difficulty: 'intermediate',
    popularity: 8,
    learningResources: 'abundant'
  },
  {
    id: 'spring',
    name: 'Spring Boot',
    category: 'バックエンド',
    language: 'Java',
    difficulty: 'intermediate',
    popularity: 8,
    learningResources: 'abundant'
  },

  // データベース
  {
    id: 'mysql',
    name: 'MySQL',
    category: 'RDBMS',
    type: 'リレーショナル',
    difficulty: 'beginner',
    popularity: 9,
    learningResources: 'abundant'
  },
  {
    id: 'postgresql',
    name: 'PostgreSQL',
    category: 'RDBMS',
    type: 'リレーショナル',
    difficulty: 'intermediate',
    popularity: 8,
    learningResources: 'abundant'
  },
  {
    id: 'mongodb',
    name: 'MongoDB',
    category: 'NoSQL',
    type: 'ドキュメント',
    difficulty: 'beginner',
    popularity: 7,
    learningResources: 'abundant'
  },

  // AI・機械学習
  {
    id: 'tensorflow',
    name: 'TensorFlow',
    category: '深層学習',
    language: 'Python',
    difficulty: 'intermediate',
    popularity: 9,
    learningResources: 'abundant'
  },
  {
    id: 'pytorch',
    name: 'PyTorch',
    category: '深層学習',
    language: 'Python',
    difficulty: 'intermediate',
    popularity: 8,
    learningResources: 'abundant'
  },
  {
    id: 'sklearn',
    name: 'scikit-learn',
    category: '機械学習',
    language: 'Python',
    difficulty: 'beginner',
    popularity: 8,
    learningResources: 'abundant'
  },

  // インフラ・DevOps
  {
    id: 'docker',
    name: 'Docker',
    category: 'コンテナ',
    type: '仮想化',
    difficulty: 'intermediate',
    popularity: 9,
    learningResources: 'abundant'
  },
  {
    id: 'kubernetes',
    name: 'Kubernetes',
    category: 'オーケストレーション',
    type: 'コンテナ管理',
    difficulty: 'advanced',
    popularity: 8,
    learningResources: 'moderate'
  },
  {
    id: 'aws',
    name: 'Amazon Web Services',
    category: 'クラウド',
    type: 'パブリッククラウド',
    difficulty: 'intermediate',
    popularity: 9,
    learningResources: 'abundant'
  },

  // モバイル開発
  {
    id: 'flutter',
    name: 'Flutter',
    category: 'モバイル',
    language: 'Dart',
    difficulty: 'intermediate',
    popularity: 7,
    learningResources: 'abundant'
  },
  {
    id: 'react_native',
    name: 'React Native',
    category: 'モバイル',
    language: 'JavaScript',
    difficulty: 'intermediate',
    popularity: 7,
    learningResources: 'abundant'
  },

  // ゲーム開発
  {
    id: 'unity',
    name: 'Unity',
    category: 'ゲーム開発',
    language: 'C#',
    difficulty: 'intermediate',
    popularity: 8,
    learningResources: 'abundant'
  },

  // 画像・映像処理
  {
    id: 'opencv',
    name: 'OpenCV',
    category: 'コンピュータビジョン',
    language: 'Python/C++',
    difficulty: 'intermediate',
    popularity: 8,
    learningResources: 'abundant'
  }
];

// フィールドユーティリティ
export const fieldUtils = {
  getFieldName: (fieldId: string): string => {
    const field = RESEARCH_FIELDS.find(f => f.id === fieldId);
    return field ? field.name : fieldId;
  },
  
  getFieldDescription: (fieldId: string): string => {
    const field = RESEARCH_FIELDS.find(f => f.id === fieldId);
    return field ? field.description : '';
  },

  getFieldsByCategory: (category: string): ResearchField[] => {
    return RESEARCH_FIELDS.filter(field => field.category === category);
  },

  getFieldsByDifficulty: (difficulty: 'beginner' | 'intermediate' | 'advanced'): ResearchField[] => {
    return RESEARCH_FIELDS.filter(field => field.difficulty === difficulty);
  },

  getHighDemandFields: (): ResearchField[] => {
    return RESEARCH_FIELDS.filter(field => field.marketDemand === 'high');
  }
};

// 言語ユーティリティ
export const languageUtils = {
  getLanguageName: (languageId: string): string => {
    const language = PROGRAMMING_LANGUAGES.find(l => l.id === languageId);
    return language ? language.name : languageId;
  },

  getLanguagesByCategory: (category: string): ProgrammingLanguage[] => {
    return PROGRAMMING_LANGUAGES.filter(lang => lang.category === category);
  },

  getBeginnerFriendlyLanguages: (): ProgrammingLanguage[] => {
    return PROGRAMMING_LANGUAGES.filter(lang => lang.difficulty === 'beginner');
  },

  getHighDemandLanguages: (): ProgrammingLanguage[] => {
    return PROGRAMMING_LANGUAGES.filter(lang => lang.marketDemand === 'high');
  }
};

// 技術スタックユーティリティ
export const techUtils = {
  getFrameworksByCategory: (category: string): TechFramework[] => {
    return TECH_FRAMEWORKS.filter(tech => tech.category === category);
  },

  getPopularFrameworks: (minPopularity: number = 7): TechFramework[] => {
    return TECH_FRAMEWORKS.filter(tech => tech.popularity >= minPopularity);
  },

  getFrameworksByLanguage: (language: string): TechFramework[] => {
    return TECH_FRAMEWORKS.filter(tech => tech.language === language);
  }
};

// APIサービス
export const apiService = {
  // 既存のAPI関数
  async evaluateCompatibility(preferences: EvaluationPreferences): Promise<EvaluationResponse> {
    try {
      const response = await api.post('/evaluate', preferences);
      return response.data;
    } catch (error) {
      console.error('Compatibility evaluation failed:', error);
      throw new Error('研究室適合度の評価に失敗しました。');
    }
  },

  async getLabs(): Promise<Lab[]> {
    try {
      const response = await api.get('/labs');
      return response.data.labs || [];
    } catch (error) {
      console.error('Failed to fetch labs:', error);
      throw new Error('研究室データの取得に失敗しました。');
    }
  },

  async getHealthStatus(): Promise<{ status: string; message: string }> {
    try {
      const response = await api.get('/health');
      return response.data;
    } catch (error) {
      console.error('Health check failed:', error);
      return { status: 'error', message: 'サーバーに接続できません。' };
    }
  },

  // 新しいAPI関数
  async getResearchFields(): Promise<ResearchField[]> {
    // 現在は静的データを返すが、将来的にはAPIから取得可能
    return Promise.resolve(RESEARCH_FIELDS);
  },

  async getProgrammingLanguages(): Promise<ProgrammingLanguage[]> {
    // 現在は静的データを返すが、将来的にはAPIから取得可能
    return Promise.resolve(PROGRAMMING_LANGUAGES);
  },

  async getTechFrameworks(): Promise<TechFramework[]> {
    // 現在は静的データを返すが、将来的にはAPIから取得可能
    return Promise.resolve(TECH_FRAMEWORKS);
  },

  async evaluateWithTechStack(
    preferences: EvaluationPreferences,
    fieldInterests: { [fieldId: string]: FieldInterest },
    techStackPreferences: TechStackPreference
  ): Promise<EvaluationResponse> {
    try {
      const enhancedPreferences = {
        ...preferences,
        research_field_interests: fieldInterests,
        tech_stack_preferences: techStackPreferences
      };

      const response = await api.post('/evaluate', enhancedPreferences);
      return response.data;
    } catch (error) {
      console.error('Enhanced compatibility evaluation failed:', error);
      throw new Error('研究室適合度の評価に失敗しました。');
    }
  }
};

// エクスポートされたインターフェース（下位互換性のため）
export interface EnhancedEvaluationPreferences extends EvaluationPreferences {
  research_field_interests?: {
    [fieldId: string]: FieldInterest;
  };
  tech_stack_preferences?: TechStackPreference;
}

// フィールドマッチング結果
export interface FieldMatchingResult {
  matched_fields: string[];
  field_scores: { [fieldId: string]: number };
  field_weight: number;
}

// フィールド分析結果
export interface FieldAnalysis {
  selected_fields_count: number;
  average_interest: number;
  primary_category: string;
  field_coverage: number;
}

// カテゴリ定数
export const FIELD_CATEGORIES = [
  '情報工学・AI',
  'Web・アプリ開発',
  '基盤技術',
  '医療・健康',
  'ビジネス・経営',
  'メディア・コンテンツ',
  'デザイン・UX',
  '組み込み・IoT',
  '新興技術',
  '先端技術',
  '理論・数学',
  'データ科学',
  'エンターテイメント',
  '生命科学',
  '環境・地球科学',
  '社会応用',
  '理論・検証'
];

export const LANGUAGE_CATEGORIES = [
  '汎用・スクリプト',
  '汎用・エンタープライズ',
  'Web・フロントエンド',
  'Web・マークアップ',
  'Web・サーバーサイド',
  'モバイル・iOS',
  'モバイル・Android',
  'モバイル・クロスプラットフォーム',
  'システム・高性能',
  'システム・低レベル',
  'システム・バックエンド',
  'システム・安全性',
  'エンタープライズ・Microsoft',
  'データ分析・統計',
  'データベース',
  '科学計算・工学',
  '科学計算・高性能',
  '関数型・学術',
  'ビッグデータ・関数型',
  'ブロックチェーン',
  '低レベル・組み込み',
  'ハードウェア設計'
];

export const TECH_CATEGORIES = [
  'フロントエンド',
  'バックエンド',
  'RDBMS',
  'NoSQL',
  '深層学習',
  '機械学習',
  'コンテナ',
  'オーケストレーション',
  'クラウド',
  'モバイル',
  'ゲーム開発',
  'コンピュータビジョン'
];

export default api;