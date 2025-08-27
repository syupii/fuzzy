import React, { useState } from 'react';
import {
  Box,
  Typography,
  Slider,
  Paper,
  Grid,
  Button,
  Card,
  CardContent,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Alert,
  Chip,
  LinearProgress,
} from '@mui/material';
import {
  ExpandMore,
  School,
  Groups,
  Psychology,
  TrendingUp,
  Schedule,
  Star,
} from '@mui/icons-material';
import { 
  EvaluationPreferences, 
  FieldInterest, 
  TechStackPreference,
  fieldUtils,
  languageUtils 
} from '../services/api';

interface EvaluationFormProps {
  onResults: (preferences: EvaluationPreferences) => void;
  fieldSelections?: { [fieldId: string]: FieldInterest };
  techStackPreferences?: TechStackPreference;
}

const EvaluationForm: React.FC<EvaluationFormProps> = ({
  onResults,
  fieldSelections,
  techStackPreferences,
}) => {
  // 評価設定の状態
  const [preferences, setPreferences] = useState<EvaluationPreferences>({
    // 基本的な研究環境
    research_intensity: 5,
    advisor_style: 5,
    team_work: 5,
    workload: 5,
    theory_practice: 5,
    research_field_match: 8, // 分野適合性は重要度高めに設定

    // 学習・成長
    skill_development: 7,
    learning_pace: 5,
    difficulty_preference: 5,

    // コミュニケーション・環境
    communication_style: 5,
    meeting_frequency: 5,
    lab_atmosphere: 5,

    // 研究アプローチ
    innovation_risk: 5,
    methodology_preference: 5,
    interdisciplinary: 5,

    // 時間・ライフスタイル
    flexibility: 6,
    evening_weekend_work: 4,

    // 重要成功要因
    publication_opportunity: 6,
    financial_support: 7,
    lab_hierarchy: 5,
    core_time_flexibility: 6,
  });

  const [expandedSections, setExpandedSections] = useState<string[]>(['basic', 'priority']);

  // 評価項目の定義
  const evaluationCategories = {
    basic: {
      title: '基本的な研究環境',
      icon: <School color="primary" />,
      criteria: {
        research_intensity: {
          label: '研究強度',
          description: 'どの程度集中的に研究に取り組みたいか',
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
      icon: <Psychology color="warning" />,
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
      title: '重要成功要因',
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

  // セクション展開/折りたたみ
  const handleSectionToggle = (section: string) => {
    setExpandedSections(prev =>
      prev.includes(section)
        ? prev.filter(s => s !== section)
        : [...prev, section]
    );
  };

  // 設定値の変更
  const handlePreferenceChange = (key: keyof EvaluationPreferences, value: number) => {
    setPreferences(prev => ({
      ...prev,
      [key]: value,
    }));
  };

  // 送信処理
  const handleSubmit = () => {
    onResults(preferences);
  };

  // 選択サマリーの作成
  const getSelectionSummary = () => {
    const selectedFields = fieldSelections ? Object.entries(fieldSelections)
      .filter(([, interest]) => interest.isSelected)
      .map(([fieldId]) => fieldUtils.getFieldName(fieldId)) : [];
    
    const selectedLanguages = techStackPreferences ? techStackPreferences.languagePreferences
      .map(langId => languageUtils.getLanguageName(langId)) : [];

    return { selectedFields, selectedLanguages };
  };

  const { selectedFields, selectedLanguages } = getSelectionSummary();

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom sx={{ mb: 3, textAlign: 'center' }}>
        ⚙️ 詳細設定・評価項目
      </Typography>

      <Alert severity="info" sx={{ mb: 3 }}>
        <Typography variant="body2">
          研究室に対する詳細な希望を1-10のスケールで設定してください。
          これまでの選択内容と合わせて、最適な研究室をマッチングします。
        </Typography>
      </Alert>

      {/* 選択内容サマリー */}
      <Paper sx={{ p: 3, mb: 4, bgcolor: '#f8f9fa' }}>
        <Typography variant="h6" gutterBottom>これまでの選択内容</Typography>
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Typography variant="subtitle2" gutterBottom>研究分野 ({selectedFields.length}項目)</Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
              {selectedFields.slice(0, 5).map(field => (
                <Chip key={field} label={field} size="small" color="primary" variant="outlined" />
              ))}
              {selectedFields.length > 5 && (
                <Chip label={`+${selectedFields.length - 5}個`} size="small" color="primary" />
              )}
            </Box>
          </Grid>
          <Grid item xs={12} md={6}>
            <Typography variant="subtitle2" gutterBottom>プログラミング言語 ({selectedLanguages.length}項目)</Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
              {selectedLanguages.slice(0, 5).map(lang => (
                <Chip key={lang} label={lang} size="small" color="secondary" variant="outlined" />
              ))}
              {selectedLanguages.length > 5 && (
                <Chip label={`+${selectedLanguages.length - 5}個`} size="small" color="secondary" />
              )}
            </Box>
          </Grid>
        </Grid>
        <Box sx={{ mt: 2 }}>
          <Typography variant="body2" color="text.secondary">
            学習意欲: {techStackPreferences?.learningWillingness || 'N/A'}/10 • 
            キャリア目標: {techStackPreferences?.careerGoals.length || 0}個選択
          </Typography>
        </Box>
      </Paper>

      {/* 評価項目設定 */}
      {Object.entries(evaluationCategories).map(([categoryKey, category]) => (
        <Accordion
          key={categoryKey}
          expanded={expandedSections.includes(categoryKey)}
          onChange={() => handleSectionToggle(categoryKey)}
          sx={{ mb: 2 }}
        >
          <AccordionSummary expandIcon={<ExpandMore />}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              {category.icon}
              <Typography variant="h6">{category.title}</Typography>
              <Chip
                label={`${Object.keys(category.criteria).length}項目`}
                size="small"
                variant="outlined"
              />
            </Box>
          </AccordionSummary>
          <AccordionDetails>
            <Grid container spacing={3}>
              {Object.entries(category.criteria).map(([criterionKey, criterion]) => {
                const key = criterionKey as keyof EvaluationPreferences;
                const value = preferences[key];
                
                return (
                  <Grid item xs={12} md={6} key={criterionKey}>
                    <Card variant="outlined" sx={{ p: 2, height: '100%' }}>
                      <Typography variant="subtitle1" gutterBottom fontWeight="bold">
                        {criterion.label}
                      </Typography>
                      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                        {criterion.description}
                      </Typography>
                      
                      <Box sx={{ mb: 2 }}>
                        <Typography variant="body2" gutterBottom>
                          現在の設定: {value}/10
                        </Typography>
                        <Slider
                          value={value}
                          onChange={(_, newValue) => handlePreferenceChange(key, newValue as number)}
                          min={1}
                          max={10}
                          step={1}
                          marks={[
                            { value: 1, label: criterion.min },
                            { value: 10, label: criterion.max }
                          ]}
                          valueLabelDisplay="auto"
                          color="primary"
                        />
                      </Box>
                      
                      <LinearProgress 
                        variant="determinate" 
                        value={value * 10} 
                        color={value >= 7 ? 'success' : value >= 4 ? 'warning' : 'error'}
                        sx={{ height: 6, borderRadius: 3 }}
                      />
                    </Card>
                  </Grid>
                );
              })}
            </Grid>
          </AccordionDetails>
        </Accordion>
      ))}

      {/* 設定完了ボタン */}
      <Box sx={{ textAlign: 'center', mt: 4 }}>
        <Button
          variant="contained"
          size="large"
          onClick={handleSubmit}
          sx={{ minWidth: 250, py: 1.5 }}
          startIcon={<Star />}
        >
          研究室マッチング実行
        </Button>
        <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
          設定内容に基づいて最適な研究室を分析・推薦します
        </Typography>
      </Box>
    </Box>
  );
};

export default EvaluationForm;