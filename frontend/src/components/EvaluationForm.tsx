// src/components/EvaluationForm.tsx - 修正版
import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Slider,
  Button,
  Grid,
  Alert,
  CircularProgress,
  Chip,
  Paper,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Divider,
} from '@mui/material';
import {
  Science,
  Psychology,
  TrendingUp,
  ExpandMore,
  School,
  Groups,
  Explore,
  Schedule,
  Article,
  AttachMoney,
  AccountTree,
  AccessTime
} from '@mui/icons-material';
import {
  apiService,
  EvaluationPreferences,
  EvaluationResponse,
  ResearchFieldInterests,
  StudentProfile,
  RESEARCH_FIELDS,
  FIELD_CATEGORIES,
  fieldUtils
} from '../services/api';

interface EvaluationFormProps {
  onResults: (results: EvaluationResponse) => void;
}

const EvaluationForm: React.FC<EvaluationFormProps> = ({ onResults }) => {
  // 20項目評価基準の初期値
  const [preferences, setPreferences] = useState<EvaluationPreferences>({
    // 基本項目（5項目）
    research_intensity: 7.0,
    advisor_style: 6.0,
    team_work: 7.0,
    workload: 6.0,
    theory_practice: 7.0,

    // 拡張項目（5項目）
    research_field_match: 8.0,
    skill_development: 7.0,
    learning_pace: 6.0,
    difficulty_preference: 7.0,
    lab_atmosphere: 7.0,

    // コミュニケーション関連（4項目）
    communication_style: 6.0,
    meeting_frequency: 6.0,
    flexibility: 7.0,
    evening_weekend_work: 5.0,

    // 研究アプローチ関連（3項目）
    innovation_risk: 6.0,
    methodology_preference: 6.0,
    interdisciplinary: 6.0,

    // 重要項目（3項目）
    publication_opportunity: 8.0,
    financial_support: 7.0,
    lab_hierarchy: 6.0,
    core_time_flexibility: 7.0,
  });

  // 研究分野の興味度
  const [fieldInterests, setFieldInterests] = useState<ResearchFieldInterests>({
    "人工知能・機械学習": 7.0,
    "画像・映像処理": 5.0,
    "コンピュータネットワーク・セキュリティ": 4.0,
    "データベース・情報システム": 5.0,
    "組込み・IoT": 4.0,
    "Webデザイン・UI/UX": 6.0,
    "デザイン・視覚表現": 5.0,
    "映像・アニメーション": 4.0,
    "コンピュータ音楽・サウンドアート": 3.0,
    "ゲーム開発・eスポーツ": 6.0,
    "VR/AR・メディアアート": 5.0
  });

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeStep, setActiveStep] = useState(0);

  // カテゴリ別の評価基準情報
  const criteriaCategories = {
    basic: {
      title: '基本的な研究環境',
      icon: <Science color="primary" />,
      criteria: {
        research_intensity: {
          label: '研究強度',
          description: '研究活動の集中度と深さ',
          leftLabel: '軽い研究',
          rightLabel: '集中研究',
          emoji: '🔬'
        },
        advisor_style: {
          label: '指導スタイル',
          description: '教授からの指導の受け方',
          leftLabel: '厳格指導',
          rightLabel: '自由指導',
          emoji: '👨‍🏫'
        },
        team_work: {
          label: 'チームワーク',
          description: '他者との協働の程度',
          leftLabel: '個人研究',
          rightLabel: 'チーム研究',
          emoji: '🤝'
        },
        workload: {
          label: 'ワークロード',
          description: '研究活動の忙しさ',
          leftLabel: '軽い負荷',
          rightLabel: '重い負荷',
          emoji: '⚡'
        },
        theory_practice: {
          label: '理論・実践バランス',
          description: '理論と実践のバランス',
          leftLabel: '理論重視',
          rightLabel: '実践重視',
          emoji: '⚖️'
        }
      }
    },
    extended: {
      title: '学習・成長環境',
      icon: <TrendingUp color="primary" />,
      criteria: {
        research_field_match: {
          label: '分野適合性',
          description: '興味と研究分野の一致度',
          leftLabel: '広い分野',
          rightLabel: '専門特化',
          emoji: '🎯'
        },
        skill_development: {
          label: 'スキル開発',
          description: '専門性と汎用性のバランス',
          leftLabel: '専門特化',
          rightLabel: '幅広いスキル',
          emoji: '📈'
        },
        learning_pace: {
          label: '学習ペース',
          description: '学習の進行速度',
          leftLabel: 'ゆっくり',
          rightLabel: '速いペース',
          emoji: '🏃'
        },
        difficulty_preference: {
          label: '難易度選好',
          description: '挑戦レベルの好み',
          leftLabel: '安全志向',
          rightLabel: '挑戦志向',
          emoji: '🎢'
        },
        lab_atmosphere: {
          label: '研究室雰囲気',
          description: '研究室の全体的な雰囲気',
          leftLabel: '静寂集中',
          rightLabel: '活発議論',
          emoji: '🌟'
        }
      }
    },
    communication: {
      title: 'コミュニケーション・時間',
      icon: <Groups color="primary" />,
      criteria: {
        communication_style: {
          label: 'コミュニケーション',
          description: '研究室での交流スタイル',
          leftLabel: '少人数密接',
          rightLabel: 'オープン交流',
          emoji: '💬'
        },
        meeting_frequency: {
          label: 'ミーティング頻度',
          description: '定期的な会議の頻度',
          leftLabel: '少ない',
          rightLabel: '頻繁',
          emoji: '📅'
        },
        flexibility: {
          label: '柔軟性',
          description: '研究時間の自由度',
          leftLabel: '固定スケジュール',
          rightLabel: '柔軟スケジュール',
          emoji: '🤸'
        },
        evening_weekend_work: {
          label: '夜間・休日作業',
          description: '時間外作業の許容度',
          leftLabel: '平日のみ',
          rightLabel: '24時間対応',
          emoji: '🌙'
        }
      }
    },
    approach: {
      title: '研究アプローチ',
      icon: <Explore color="primary" />,
      criteria: {
        innovation_risk: {
          label: '革新性・リスク許容度',
          description: '新しい手法への挑戦度',
          leftLabel: '安全手法',
          rightLabel: '革新手法',
          emoji: '🚀'
        },
        methodology_preference: {
          label: '手法選好',
          description: '研究手法の好み',
          leftLabel: '確立手法',
          rightLabel: '新手法',
          emoji: '🔧'
        },
        interdisciplinary: {
          label: '学際性',
          description: '他分野との連携の程度',
          leftLabel: '単一分野',
          rightLabel: '学際連携',
          emoji: '🌐'
        }
      }
    },
    priority: {
      title: '重要項目（学生調査結果）',
      icon: <AttachMoney color="secondary" />,
      criteria: {
        publication_opportunity: {
          label: '論文発表機会',
          description: '研究成果の論文化機会',
          leftLabel: '少ない機会',
          rightLabel: '豊富な機会',
          emoji: '📝'
        },
        financial_support: {
          label: '経済支援',
          description: '研究資金や奨学金サポート',
          leftLabel: '最小限',
          rightLabel: '充実',
          emoji: '💰'
        },
        lab_hierarchy: {
          label: '研究室階層',
          description: '研究室内の上下関係',
          leftLabel: '厳格階層',
          rightLabel: 'フラット',
          emoji: '👥'
        },
        core_time_flexibility: {
          label: 'コアタイム柔軟性',
          description: '必須出席時間の柔軟さ',
          leftLabel: '固定時間',
          rightLabel: '自由出席',
          emoji: '⏰'
        }
      }
    }
  };

  const handlePreferenceChange = (key: keyof EvaluationPreferences, value: number) => {
    setPreferences(prev => ({ ...prev, [key]: value }));
  };

  const handleFieldInterestChange = (fieldName: string, value: number) => {
    setFieldInterests(prev => ({ ...prev, [fieldName]: value }));
  };

  const handleEvaluate = async () => {
    setLoading(true);
    setError(null);

    try {
      const studentProfile: StudentProfile = {
        evaluation_criteria: preferences,
        field_interests: fieldInterests
      };

      const response = await apiService.evaluateCompatibility(studentProfile);
      onResults(response);
    } catch (err: any) {
      setError(err.response?.data?.error || '評価に失敗しました。サーバーが起動しているか確認してください。');
    } finally {
      setLoading(false);
    }
  };

  const handleOptimizeWithGA = async () => {
    setLoading(true);
    setError(null);

    try {
      const studentProfile: StudentProfile = {
        evaluation_criteria: preferences,
        field_interests: fieldInterests
      };

      const response = await apiService.optimizeWithGeneticAlgorithm(studentProfile);
      onResults(response);
    } catch (err: any) {
      setError(err.response?.data?.error || '遺伝的アルゴリズム最適化に失敗しました。');
    } finally {
      setLoading(false);
    }
  };

  const loadDemoProfile = async () => {
    try {
      const profile = await apiService.getDemoProfile();
      setPreferences(profile.evaluation_criteria);
      setFieldInterests(profile.field_interests);
    } catch (err) {
      setError('デモプロフィールの読み込みに失敗しました。');
    }
  };

  const resetToDefaults = () => {
    setPreferences({
      research_intensity: 7.0,
      advisor_style: 6.0,
      team_work: 7.0,
      workload: 6.0,
      theory_practice: 7.0,
      research_field_match: 8.0,
      skill_development: 7.0,
      learning_pace: 6.0,
      difficulty_preference: 7.0,
      lab_atmosphere: 7.0,
      communication_style: 6.0,
      meeting_frequency: 6.0,
      flexibility: 7.0,
      evening_weekend_work: 5.0,
      innovation_risk: 6.0,
      methodology_preference: 6.0,
      interdisciplinary: 6.0,
      publication_opportunity: 8.0,
      financial_support: 7.0,
      lab_hierarchy: 6.0,
      core_time_flexibility: 7.0,
    });

    setFieldInterests({
      "人工知能・機械学習": 7.0,
      "画像・映像処理": 5.0,
      "コンピュータネットワーク・セキュリティ": 4.0,
      "データベース・情報システム": 5.0,
      "組込み・IoT": 4.0,
      "Webデザイン・UI/UX": 6.0,
      "デザイン・視覚表現": 5.0,
      "映像・アニメーション": 4.0,
      "コンピュータ音楽・サウンドアート": 3.0,
      "ゲーム開発・eスポーツ": 6.0,
      "VR/AR・メディアアート": 5.0
    });
  };

  const getScoreColor = (score: number): 'success' | 'warning' | 'error' => {
    if (score >= 8) return 'success';
    if (score >= 6) return 'warning';
    return 'error';
  };

  return (
    <Card elevation={3} sx={{ mb: 4 }}>
      <CardContent>
        <Box sx={{ textAlign: 'center', mb: 4 }}>
          <Science color="primary" sx={{ fontSize: 48, mb: 2 }} />
          <Typography variant="h4" component="h2" gutterBottom color="primary">
            拡張版研究室適合度評価
          </Typography>
          <Typography variant="body1" color="text.secondary">
            21項目の詳細な評価基準であなたの希望を分析します（1-10スケール）
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
            実際の学生調査結果に基づく重要項目を含む包括的評価
          </Typography>
        </Box>

        {error && (
          <Alert severity="error" sx={{ mb: 3 }}>
            {error}
          </Alert>
        )}

        {/* コントロールボタン */}
        <Box sx={{ display: 'flex', gap: 2, mb: 3, justifyContent: 'center' }}>
          <Button variant="outlined" onClick={loadDemoProfile} size="small">
            デモデータ読込
          </Button>
          <Button variant="outlined" onClick={resetToDefaults} size="small">
            初期値に戻す
          </Button>
        </Box>

        {/* カテゴリ別評価項目 */}
        {Object.entries(criteriaCategories).map(([categoryKey, category]) => (
          <Accordion key={categoryKey} defaultExpanded={categoryKey === 'priority'}>
            <AccordionSummary
              expandIcon={<ExpandMore />}
              aria-controls={`${categoryKey}-content`}
              id={`${categoryKey}-header`}
            >
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                {category.icon}
                <Typography variant="h6">{category.title}</Typography>
                <Chip
                  label={`${Object.keys(category.criteria).length}項目`}
                  size="small"
                  color={categoryKey === 'priority' ? 'secondary' : 'primary'}
                  variant="outlined"
                />
              </Box>
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={3}>
                {Object.entries(category.criteria).map(([key, criterion]) => (
                  <Grid item xs={12} md={6} key={key}>
                    <Paper elevation={1} sx={{ p: 2, height: '100%' }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                        <Typography sx={{ fontSize: '1.5rem' }}>
                          {criterion.emoji}
                        </Typography>
                        <Box>
                          <Typography variant="h6" gutterBottom>
                            {criterion.label}
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            {criterion.description}
                          </Typography>
                        </Box>
                      </Box>

                      <Slider
                        value={preferences[key as keyof EvaluationPreferences]}
                        onChange={(_, value) => handlePreferenceChange(key as keyof EvaluationPreferences, value as number)}
                        min={1}
                        max={10}
                        step={0.5}
                        marks
                        valueLabelDisplay="on"
                        color={getScoreColor(preferences[key as keyof EvaluationPreferences])}
                        sx={{ mb: 2 }}
                      />

                      <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                        <Typography variant="caption" color="text.secondary">
                          {criterion.leftLabel}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {criterion.rightLabel}
                        </Typography>
                      </Box>
                    </Paper>
                  </Grid>
                ))}
              </Grid>
            </AccordionDetails>
          </Accordion>
        ))}

        {/* 研究分野興味度 */}
        <Accordion defaultExpanded>
          <AccordionSummary expandIcon={<ExpandMore />}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <Science color="primary" />
              <Typography variant="h6">研究分野興味度</Typography>
              <Chip
                label={`${Object.keys(fieldInterests).length}分野`}
                size="small"
                color="primary"
                variant="outlined"
              />
            </Box>
          </AccordionSummary>
          <AccordionDetails>
            <Grid container spacing={2}>
              {FIELD_CATEGORIES.map(category => {
                const fields = fieldUtils.getFieldsByCategory(category);
                return (
                  <Grid item xs={12} key={category}>
                    <Typography variant="h6" gutterBottom color="primary">
                      {category}
                    </Typography>
                    <Grid container spacing={2}>
                      {fields.map(field => (
                        <Grid item xs={12} md={6} key={field.id}>
                          <Paper elevation={1} sx={{ p: 2 }}>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                              <Typography sx={{ fontSize: '1.2rem' }}>
                                {field.icon}
                              </Typography>
                              <Typography variant="subtitle1">
                                {field.name}
                              </Typography>
                            </Box>
                            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                              {field.description}
                            </Typography>
                            <Slider
                              value={fieldInterests[field.name] || 5}
                              onChange={(_, value) => handleFieldInterestChange(field.name, value as number)}
                              min={1}
                              max={10}
                              step={0.5}
                              marks
                              valueLabelDisplay="on"
                              color={getScoreColor(fieldInterests[field.name] || 5)}
                            />
                          </Paper>
                        </Grid>
                      ))}
                    </Grid>
                  </Grid>
                );
              })}
            </Grid>
          </AccordionDetails>
        </Accordion>

        {/* 実行ボタン */}
        <Divider sx={{ my: 4 }} />
        <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center' }}>
          <Button
            variant="contained"
            onClick={handleEvaluate}
            disabled={loading}
            size="large"
            startIcon={loading ? <CircularProgress size={20} /> : <Psychology />}
            sx={{ minWidth: 200 }}
          >
            {loading ? '評価中...' : '適合度評価を実行'}
          </Button>

          <Button
            variant="outlined"
            onClick={handleOptimizeWithGA}
            disabled={loading}
            size="large"
            startIcon={loading ? <CircularProgress size={20} /> : <Psychology />}
            sx={{ minWidth: 200 }}
          >
            {loading ? '最適化中...' : '遺伝的アルゴリズム最適化'}
          </Button>
        </Box>

        {/* 評価サマリー */}
        <Paper elevation={2} sx={{ mt: 3, p: 2, bgcolor: 'grey.50' }}>
          <Typography variant="h6" gutterBottom>
            現在の設定サマリー
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle2" gutterBottom>
                高優先度項目 (8.0以上)
              </Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                {Object.entries(preferences)
                  .filter(([key, value]) => value >= 8.0)
                  .map(([key, value]) => {
                    const criterion = Object.values(criteriaCategories)
                      .flatMap(cat => Object.entries(cat.criteria))
                      .find(([k]) => k === key);
                    return (
                      <Chip
                        key={key}
                        label={`${criterion ? criterion[1].emoji : ''} ${criterion ? criterion[1].label : key}: ${value}`}
                        color="success"
                        size="small"
                      />
                    );
                  })}
              </Box>
            </Grid>

            <Grid item xs={12} md={6}>
              <Typography variant="subtitle2" gutterBottom>
                関心分野 (7.0以上)
              </Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                {Object.entries(fieldInterests)
                  .filter(([field, interest]) => (interest as number) >= 7.0)
                  .sort(([, a], [, b]) => (b as number) - (a as number))
                  .map(([field, interest]) => {
                    const fieldInfo = RESEARCH_FIELDS.find(f => f.name === field);
                    return (
                      <Chip
                        key={field}
                        label={`${fieldInfo?.icon || ''} ${field}: ${interest}`}
                        color={getScoreColor(interest as number)}
                        size="small"
                      />
                    );
                  })}
              </Box>
            </Grid>
          </Grid>
          <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
            設定完了項目: {Object.keys(preferences).length + Object.keys(fieldInterests).length}項目 |
            平均評価値: {(Object.values(preferences).reduce((a, b) => a + b, 0) / Object.keys(preferences).length).toFixed(1)} |
            関心分野数: {Object.values(fieldInterests).filter(v => (v as number) >= 6).length}分野
          </Typography>
        </Paper>
      </CardContent>
    </Card>
  );
};

export default EvaluationForm;