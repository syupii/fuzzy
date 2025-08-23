-- 20項目対応データベーススキーマ
-- SQLiteで実行してください

-- 1. labs テーブル（20項目完全対応）
CREATE TABLE labs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    professor TEXT NOT NULL,
    research_area TEXT NOT NULL,
    description TEXT,
    
    -- 20項目のファジィ特徴量（1-10スケール）
    -- 基本的な研究環境（6項目）
    research_intensity REAL DEFAULT 5.0,
    advisor_style REAL DEFAULT 5.0,
    team_work REAL DEFAULT 5.0,
    workload REAL DEFAULT 5.0,
    theory_practice REAL DEFAULT 5.0,
    research_field_match REAL DEFAULT 5.0,
    
    -- 学習・成長関連（3項目）
    skill_development REAL DEFAULT 5.0,
    learning_pace REAL DEFAULT 5.0,
    difficulty_preference REAL DEFAULT 5.0,
    
    -- コミュニケーション・環境関連（3項目）
    communication_style REAL DEFAULT 5.0,
    meeting_frequency REAL DEFAULT 5.0,
    lab_atmosphere REAL DEFAULT 5.0,
    
    -- 研究アプローチ関連（3項目）
    innovation_risk REAL DEFAULT 5.0,
    methodology_preference REAL DEFAULT 5.0,
    interdisciplinary REAL DEFAULT 5.0,
    
    -- 時間・ライフスタイル関連（2項目）
    flexibility REAL DEFAULT 5.0,
    evening_weekend_work REAL DEFAULT 5.0,
    
    -- 調査結果に基づく追加項目（3項目）
    publication_opportunity REAL DEFAULT 5.0,
    financial_support REAL DEFAULT 5.0,
    lab_hierarchy REAL DEFAULT 5.0,
    core_time_flexibility REAL DEFAULT 5.0,
    
    -- メタデータ
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT 1
);

-- 2. evaluations テーブル（20項目対応）
CREATE TABLE evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    
    -- ユーザー入力データ（20項目すべて）
    research_intensity REAL NOT NULL,
    advisor_style REAL NOT NULL,
    team_work REAL NOT NULL,
    workload REAL NOT NULL,
    theory_practice REAL NOT NULL,
    research_field_match REAL NOT NULL,
    skill_development REAL NOT NULL,
    learning_pace REAL NOT NULL,
    difficulty_preference REAL NOT NULL,
    communication_style REAL NOT NULL,
    meeting_frequency REAL NOT NULL,
    lab_atmosphere REAL NOT NULL,
    innovation_risk REAL NOT NULL,
    methodology_preference REAL NOT NULL,
    interdisciplinary REAL NOT NULL,
    flexibility REAL NOT NULL,
    evening_weekend_work REAL NOT NULL,
    publication_opportunity REAL NOT NULL,
    financial_support REAL NOT NULL,
    lab_hierarchy REAL NOT NULL,
    core_time_flexibility REAL NOT NULL,
    
    -- 結果データ
    user_preferences TEXT,
    evaluation_count INTEGER,
    avg_score REAL,
    best_lab_id INTEGER,
    engine_used TEXT,
    
    -- メタデータ
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 3. system_config テーブル
CREATE TABLE system_config (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    config_key TEXT NOT NULL UNIQUE,
    config_value TEXT,
    config_type TEXT DEFAULT 'string',
    description TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 4. genetic_individuals テーブル
CREATE TABLE genetic_individuals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    individual_id TEXT NOT NULL UNIQUE,
    generation INTEGER NOT NULL,
    genome_data TEXT,
    accuracy REAL,
    simplicity REAL,
    interpretability REAL,
    generalization REAL,
    validity REAL,
    overall_fitness REAL,
    parent1_id TEXT,
    parent2_id TEXT,
    model_complexity INTEGER,
    tree_depth INTEGER,
    evaluation_time REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 5. decision_paths テーブル
CREATE TABLE decision_paths (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    evaluation_id INTEGER,
    step_order INTEGER,
    criterion TEXT,
    threshold REAL,
    user_value REAL,
    lab_value REAL,
    decision_result TEXT,
    criterion_weight REAL,
    confidence REAL,
    rule_explanation TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (evaluation_id) REFERENCES evaluations (id)
);

-- 6. optimization_runs テーブル
CREATE TABLE optimization_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL UNIQUE,
    population_size INTEGER,
    generations INTEGER,
    mutation_rate REAL,
    crossover_rate REAL,
    max_depth INTEGER,
    tournament_size INTEGER,
    training_samples INTEGER,
    test_samples INTEGER,
    feature_names TEXT,
    target_column TEXT,
    best_fitness REAL,
    best_individual_id TEXT,
    convergence_generation INTEGER,
    final_diversity REAL,
    fitness_history TEXT,
    diversity_history TEXT,
    execution_time REAL,
    status TEXT DEFAULT 'running',
    description TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    completed_at DATETIME
);

-- 7. model_registry テーブル
CREATE TABLE model_registry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT NOT NULL UNIQUE,
    model_name TEXT,
    model_type TEXT,
    version TEXT,
    model_filepath TEXT,
    result_filepath TEXT,
    file_size_bytes INTEGER,
    checksum TEXT,
    best_fitness REAL,
    model_complexity INTEGER,
    validation_accuracy REAL,
    test_accuracy REAL,
    usage_count INTEGER DEFAULT 0,
    last_used_at DATETIME,
    is_active BOOLEAN DEFAULT 1,
    is_production_ready BOOLEAN DEFAULT 0,
    description TEXT,
    tags TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);