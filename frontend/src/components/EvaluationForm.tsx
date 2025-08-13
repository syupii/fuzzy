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
import { apiService, EvaluationPreferences, EvaluationResponse } from '../services/api';

interface EvaluationFormProps {
  onResults: (results: EvaluationResponse) => void;
}

const EvaluationForm: React.FC<EvaluationFormProps> = ({ onResults }) => {
  const [preferences, setPreferences] = useState<EvaluationPreferences>({
    // 既存項目
    research_intensity: 7.0,
    advisor_style: 6.0,
    team_work: 7.0,
    workload: 6.0,
    theory_practice: 7.0,
    
    // 分野適合性（元からあった重要項目）
    research_field_match: 8.0,
    
    // 学習・成長関連
    skill_development: 6.0,
    learning_pace: 6.0,
    difficulty_preference: 7.0,
    
    // コミュニケーション・環境関連
    communication_style: 6.0,
    meeting_frequency: 6.0,
    lab_atmosphere: 7.0,
    
    // 研究アプローチ関連
    innovation_risk: 6.0,
    methodology_preference: 6.0,
    interdisciplinary: 6.0,
    
    // 時間・ライフスタイル関連
    flexibility: 7.0,
    evening_weekend_work: 5.0,
    
    // 調査結果に基づく追加項目（最優先）
    publication_opportunity: 8.0,
    financial_support: 7.0,
    lab_hierarchy: 6.0,
    core_time_flexibility: 7.0,
  });

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // カテゴリ別の評価基準情報
  const criteriaCategories = {
    basic: {
      title: '基本的な研究環境',
      icon: <Science color="primary" />,
      criteria: {
        research_intensity: {
          label: '研究強度',
          description: '研究にどれだけ集中したいか',
          min: '基礎的',
          max: '最先端',
        },
        advisor_style: {
          label: '指導スタイル',
          description: '希望する指導の方法',
          min: '厳格',
          max: '自由',
        },
        team_work: {
          label: 'チームワーク',
          description: '研究での協働の度合い',
          min: '個人研究',
          max: 'チーム研究',
        },
        workload: {
          label: 'ワークロード',
          description: '研究の負荷・忙しさ',
          min: '軽め',
          max: '重め',
        },
        theory_practice: {
          label: '理論・実践バランス',
          description: '理論と実践のどちらを重視するか',
          min: '理論重視',
          max: '実践重視',
        },
        research_field_match: {
          label: '研究分野適合性',
          description: '自分の興味と研究分野のマッチ度',
          min: '分野外も可',
          max: '完全一致重視',
        }
      }
    },
    
    learning: {
      title: '学習・成長',
      icon: <School color="secondary" />,
      criteria: {
        skill_development: {
          label: 'スキル開発',
          description: '身につけたいスキルの範囲',
          min: '専門特化',
          max: '幅広いスキル',
        },
        learning_pace: {
          label: '学習ペース',
          description: '希望する学習の進行速度',
          min: 'じっくり型',
          max: '高速習得型',
        },
        difficulty_preference: {
          label: '難易度志向',
          description: '取り組みたい課題の難易度',
          min: '安定した課題',
          max: '挑戦的課題',
        }
      }
    },
    
    communication: {
      title: 'コミュニケーション・環境',
      icon: <Groups color="success" />,
      criteria: {
        communication_style: {
          label: 'コミュニケーション',
          description: '研究室内での交流スタイル',
          min: '少人数密接',
          max: 'オープン交流',
        },
        meeting_frequency: {
          label: 'ミーティング頻度',
          description: '相談・報告の頻度',
          min: '必要最小限',
          max: '頻繁な相談',
        },
        lab_atmosphere: {
          label: '研究室雰囲気',
          description: '研究環境の雰囲気',
          min: '静寂集中',
          max: '活発議論',
        }
      }
    },
    
    approach: {
      title: '研究アプローチ',
      icon: <Explore color="warning" />,
      criteria: {
        innovation_risk: {
          label: '革新性リスク',
          description: '研究の革新性とリスクのバランス',
          min: '確実な成果',
          max: '革新的挑戦',
        },
        methodology_preference: {
          label: '手法志向',
          description: '研究手法の選択傾向',
          min: '伝統的手法',
          max: '新しい手法',
        },
        interdisciplinary: {
          label: '学際性',
          description: '分野を超えた研究への関心',
          min: '専門特化',
          max: '分野横断',
        }
      }
    },
    
    lifestyle: {
      title: '時間・ライフスタイル',
      icon: <Schedule color="info" />,
      criteria: {
        flexibility: {
          label: '時間の柔軟性',
          description: 'スケジュールの自由度',
          min: '規則正しい',
          max: '自由なスケジュール',
        },
        evening_weekend_work: {
          label: '時間外研究',
          description: '夜間・休日の研究活動',
          min: '平日のみ',
          max: '夜間・休日も',
        }
      }
    },
    
    priority: {
      title: '重要成功要因（調査結果重視項目）',
      icon: <TrendingUp color="error" />,
      criteria: {
        publication_opportunity: {
          label: '論文執筆機会',
          description: '在学中に論文著者になれる可能性',
          min: '執筆機会少ない',
          max: '積極的に著者',
        },
        financial_support: {
          label: '経済的支援',
          description: '研究費用・経済面でのサポート',
          min: '自己負担多い',
          max: '研究費潤沢',
        },
        lab_hierarchy: {
          label: '研究室上下関係',
          description: '研究室内の人間関係の構造',
          min: '厳格な上下関係',
          max: 'フラットな関係',
        },
        core_time_flexibility: {
          label: 'コアタイム柔軟性',
          description: '研究室での必須滞在時間の自由度',
          min: '厳格な時間管理',
          max: '自由度高い',
        }
      }
    }
  };

  const handleSliderChange = (criterion: keyof EvaluationPreferences) => (
    event: Event,
    newValue: number | number[]
  ) => {
    setPreferences({
      ...preferences,
      [criterion]: newValue as number,
    });
  };

  const handleEvaluate = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await apiService.evaluateCompatibility(preferences);
      onResults(response);
      
      // セッションIDを保存
      if (response.summary.session_id) {
        localStorage.setItem('fdtlss_session_id', response.summary.session_id);
      }
    } catch (err: any) {
      setError(err.response?.data?.error || '評価に失敗しました。サーバーが起動しているか確認してください。');
    } finally {
      setLoading(false);
    }
  };

  const loadDemoData = async () => {
    try {
      const response = await apiService.getDemoData();
      setPreferences(response.demo_preferences);
    } catch (err) {
      console.error('Demo data load failed:', err);
    }
  };

  const getScoreColor = (value: number): 'success' | 'warning' | 'info' | 'error' => {
    if (value >= 8) return 'success';
    if (value >= 6) return 'warning';
    if (value >= 4) return 'info';
    return 'error';
  };

  const resetToDefaults = () => {
    setPreferences({
      research_intensity: 7.0,
      advisor_style: 6.0,
      team_work: 7.0,
      workload: 6.0,
      theory_practice: 7.0,
      research_field_match: 8.0,
      skill_development: 6.0,
      learning_pace: 6.0,
      difficulty_preference: 7.0,
      communication_style: 6.0,
      meeting_frequency: 6.0,
      lab_atmosphere: 7.0,
      innovation_risk: 6.0,
      methodology_preference: 6.0,
      interdisciplinary: 6.0,
      flexibility: 7.0,
      evening_weekend_work: 5.0,
      publication_opportunity: 8.0,
      financial_support: 7.0,
      lab_hierarchy: 6.0,
      core_time_flexibility: 7.0,
    });
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
            20項目の詳細な評価基準であなたの希望を分析します（1-10スケール）
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
          <Button variant="outlined" onClick={loadDemoData} size="small">
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
                  color={categoryKey === 'priority' ? 'error' : 'default'}
                />
              </Box>
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={3}>
                {Object.entries(category.criteria).map(([key, info]) => (
                  <Grid item xs={12} md={6} key={key}>
                    <Paper sx={{ p: 3, height: '100%', position: 'relative' }}>
                      {categoryKey === 'priority' && (
                        <Chip
                          label="重要"
                          color="error"
                          size="small"
                          sx={{ position: 'absolute', top: 8, right: 8 }}
                        />
                      )}
                      
                      <Typography variant="h6" gutterBottom>
                        {info.label}
                      </Typography>
                      
                      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                        {info.description}
                      </Typography>
                      
                      <Box sx={{ px: 2, mb: 2 }}>
                        <Slider
                          value={preferences[key as keyof EvaluationPreferences]}
                          onChange={handleSliderChange(key as keyof EvaluationPreferences)}
                          min={1}
                          max={10}
                          step={0.5}
                          marks
                          valueLabelDisplay="on"
                          sx={{ mb: 1 }}
                        />
                      </Box>
                      
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                        <Typography variant="caption" color="text.secondary">
                          {info.min}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {info.max}
                        </Typography>
                      </Box>
                      
                      <Box sx={{ textAlign: 'center' }}>
                        <Chip
                          label={`現在値: ${preferences[key as keyof EvaluationPreferences]}`}
                          color={getScoreColor(preferences[key as keyof EvaluationPreferences])}
                          variant="outlined"
                          size="small"
                        />
                      </Box>
                    </Paper>
                  </Grid>
                ))}
              </Grid>
            </AccordionDetails>
          </Accordion>
        ))}

        <Divider sx={{ my: 4 }} />

        {/* 評価実行ボタン */}
        <Box sx={{ textAlign: 'center' }}>
          <Button
            variant="contained"
            size="large"
            onClick={handleEvaluate}
            disabled={loading}
            startIcon={loading ? <CircularProgress size={20} /> : <Psychology />}
            sx={{ px: 6, py: 2, fontSize: '1.1rem' }}
          >
            {loading ? '評価中...' : '20項目で適合度を評価'}
          </Button>
          
          <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
            ファジィ論理アルゴリズムによる高精度マッチング
          </Typography>
        </Box>
      </CardContent>
    </Card>
  );
};

export default EvaluationForm;