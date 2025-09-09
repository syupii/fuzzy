// frontend/src/components/EvaluationForm.tsx - チェックボックス式ユーザビリティ向上版
import React, { useState } from 'react';
import {
  Box,
  Card,
  Grid,
  Typography,
  Slider,
  Button,
  Alert,
  CircularProgress,
  Chip,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  FormGroup,
  FormControlLabel,
  Checkbox,
  Paper,
  Divider
} from '@mui/material';
import { ExpandMore, CheckCircle, RadioButtonUnchecked } from '@mui/icons-material';
import {
  EvaluationPreferences,
  apiService,
  EvaluationResponse,
  CRITERIA_INFO,
  RESEARCH_FIELDS,
  validateEvaluationPreferences
} from '../services/api';

// プロパティ名を修正（onResults → onEvaluationComplete）
interface Props {
  onResults?: (response: EvaluationResponse) => void;
  onEvaluationComplete?: (response: EvaluationResponse) => void;
  onError: (error: string) => void;
}

// チェックボックス式研究分野選択
interface ResearchFieldSelection {
  [key: string]: boolean;
}

export const EvaluationForm: React.FC<Props> = ({
  onResults,
  onEvaluationComplete,
  onError
}) => {
  // 13項目完全対応の初期状態
  const [preferences, setPreferences] = useState<EvaluationPreferences>({
    // 基本項目（5項目）
    research_intensity: 5,
    advisor_style: 5,
    team_work: 5,
    workload: 5,
    theory_practice: 5,
    // 拡張項目（5項目）
    research_field_match: 5,
    skill_development: 5,
    lab_atmosphere: 5,
    flexibility: 5,
    publication_opportunity: 5,
    // 特殊項目（3項目）
    interdisciplinary: 5,
    communication_style: 5,
    innovation_risk: 5
  });

  // チェックボックス式研究分野選択
  const [selectedFields, setSelectedFields] = useState<ResearchFieldSelection>({});
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // 評価基準のカテゴリ分類
  const criteriaCategories = {
    basic: {
      title: '基本評価基準',
      description: 'マッチング精度に大きく影響する基本項目',
      criteria: ['research_intensity', 'advisor_style', 'team_work', 'workload', 'theory_practice'],
      color: 'primary' as const
    },
    extended: {
      title: '拡張評価基準',
      description: '研究環境や機会に関する項目',
      criteria: ['research_field_match', 'skill_development', 'lab_atmosphere', 'flexibility', 'publication_opportunity'],
      color: 'secondary' as const
    },
    special: {
      title: '特殊評価基準',
      description: '研究の特性や挑戦度に関する項目',
      criteria: ['interdisciplinary', 'communication_style', 'innovation_risk'],
      color: 'success' as const
    }
  };

  // 研究分野のカテゴリ分類（20分野・4カテゴリ対応）
  const fieldCategories = {
    technology: {
      title: '🔧 テクノロジー・システム分野',
      description: 'AI、ネットワーク、データベース、教育システム、自然科学など（12分野）',
      color: 'primary' as const,
      fields: RESEARCH_FIELDS.filter(f => f.category === 'technology')
    },
    creative: {
      title: '🎨 クリエイティブ分野',
      description: 'デザイン、映像、音楽、アートなど（4分野）',
      color: 'secondary' as const,
      fields: RESEARCH_FIELDS.filter(f => f.category === 'creative')
    },
    entertainment: {
      title: '🎮 エンターテイメント分野',
      description: 'ゲーム、VR/AR、メディアアートなど（2分野）',
      color: 'success' as const,
      fields: RESEARCH_FIELDS.filter(f => f.category === 'entertainment')
    },
    humanities: {
      title: '🏛️ 人文・社会・体育分野',
      description: '哲学、環境行動学、スポーツ科学など（2分野）',
      color: 'warning' as const,
      fields: RESEARCH_FIELDS.filter(f => f.category === 'humanities')
    }
  };

  // 評価基準の変更ハンドラ
  const handlePreferenceChange = (key: keyof EvaluationPreferences, value: number) => {
    setPreferences(prev => ({
      ...prev,
      [key]: value
    }));
    setError(null);
  };

  // 研究分野チェックボックス変更ハンドラ
  const handleFieldSelection = (fieldId: string, checked: boolean) => {
    setSelectedFields(prev => ({
      ...prev,
      [fieldId]: checked
    }));
    setError(null);
  };

  // 全選択/全解除
  const handleSelectAllFields = (category: string, selectAll: boolean) => {
    const categoryFields = fieldCategories[category as keyof typeof fieldCategories].fields;
    setSelectedFields(prev => {
      const newSelection = { ...prev };
      categoryFields.forEach(field => {
        newSelection[field.id] = selectAll;
      });
      return newSelection;
    });
  };

  // 選択された分野数をカウント
  const getSelectedFieldsCount = () => {
    return Object.values(selectedFields).filter(Boolean).length;
  };

  // 評価実行
  const handleEvaluation = async () => {
    setIsLoading(true);
    setError(null);

    try {
      // バリデーション
      const validationErrors = validateEvaluationPreferences(preferences);
      if (validationErrors.length > 0) {
        throw new Error(`入力エラー: ${validationErrors.join(', ')}`);
      }

      // 研究分野選択の確認
      const selectedFieldsList = Object.keys(selectedFields).filter(key => selectedFields[key]);
      if (selectedFieldsList.length === 0) {
        throw new Error('最低1つの研究分野を選択してください');
      }

      // 選択された分野をスコア形式に変換（APIとの互換性維持）
      const fieldInterests: { [key: string]: number } = {};
      selectedFieldsList.forEach(fieldId => {
        fieldInterests[fieldId] = 8.0; // 選択された分野は高いスコア
      });

      console.log('🚀 評価開始:', {
        preferences,
        selectedFields: selectedFieldsList,
        fieldInterests
      });

      // 一時的にfield_interestsを追加してAPIコール
      const enhancedProfile = {
        ...preferences,
        field_interests: fieldInterests
      };

      const response = await apiService.evaluateLabs(enhancedProfile);

      console.log('✅ 評価完了:', response);

      // 両方のコールバックをサポート
      if (onResults) {
        onResults(response);
      }
      if (onEvaluationComplete) {
        onEvaluationComplete(response);
      }

    } catch (err: any) {
      const errorMessage = err.message || '評価処理でエラーが発生しました';
      setError(errorMessage);
      onError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  // デモデータ読み込み
  const handleLoadDemo = async () => {
    try {
      const demoProfile = await apiService.getDemoProfile();
      setPreferences(demoProfile.evaluation_criteria);

      // デモの研究分野を選択状態に設定
      const demoFieldSelection: ResearchFieldSelection = {};
      Object.keys(demoProfile.field_interests).forEach(field => {
        if (demoProfile.field_interests[field] > 5) { // スコア5以上を選択とみなす
          demoFieldSelection[field] = true;
        }
      });
      setSelectedFields(demoFieldSelection);

      setError(null);
    } catch (err) {
      setError('デモデータの読み込みに失敗しました');
    }
  };

  // 評価基準カテゴリのレンダリング
  const renderCriteriaCategory = (categoryKey: string, category: any) => (
    <Accordion key={categoryKey} defaultExpanded={true}>
      <AccordionSummary expandIcon={<ExpandMore />}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Chip label={category.title} color={category.color} size="small" />
          <Typography variant="h6">{category.title}</Typography>
        </Box>
      </AccordionSummary>
      <AccordionDetails>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          {category.description}
        </Typography>
        <Grid container spacing={3}>
          {category.criteria.map((key: string) => {
            const info = CRITERIA_INFO[key as keyof typeof CRITERIA_INFO];
            return (
              <Grid item xs={12} md={6} key={key}>
                <Card variant="outlined" sx={{ p: 2 }}>
                  <Typography variant="subtitle1" gutterBottom>
                    {info.name}
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                    {info.description}
                  </Typography>
                  <Box sx={{ px: 2 }}>
                    <Slider
                      value={preferences[key as keyof EvaluationPreferences]}
                      onChange={(_, value) => handlePreferenceChange(key as keyof EvaluationPreferences, value as number)}
                      min={1}
                      max={10}
                      step={1}
                      marks
                      valueLabelDisplay="on"
                      color={category.color}
                    />
                  </Box>
                  <Typography variant="caption" color="text.secondary">
                    {info.range}
                  </Typography>
                </Card>
              </Grid>
            );
          })}
        </Grid>
      </AccordionDetails>
    </Accordion>
  );

  // 研究分野チェックボックスのレンダリング
  const renderFieldCategory = (categoryKey: string, category: any) => (
    <Card key={categoryKey} sx={{ mb: 3 }}>
      <Box sx={{ p: 2, bgcolor: `${category.color}.50` }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Chip label={category.title} color={category.color} />
            <Typography variant="h6">{category.title}</Typography>
          </Box>
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Button
              size="small"
              onClick={() => handleSelectAllFields(categoryKey, true)}
              color={category.color}
            >
              全選択
            </Button>
            <Button
              size="small"
              onClick={() => handleSelectAllFields(categoryKey, false)}
              variant="outlined"
              color={category.color}
            >
              全解除
            </Button>
          </Box>
        </Box>

        <FormGroup>
          <Grid container spacing={1}>
            {category.fields.map((field: any) => (
              <Grid item xs={12} sm={6} md={4} key={field.id}>
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={selectedFields[field.id] || false}
                      onChange={(e) => handleFieldSelection(field.id, e.target.checked)}
                      color={category.color}
                      icon={<RadioButtonUnchecked />}
                      checkedIcon={<CheckCircle />}
                    />
                  }
                  label={
                    <Box>
                      <Typography variant="body2" fontWeight="medium">
                        {field.name}
                      </Typography>
                      {field.description && (
                        <Typography variant="caption" color="text.secondary">
                          {field.description}
                        </Typography>
                      )}
                    </Box>
                  }
                  sx={{
                    m: 0,
                    display: 'flex',
                    alignItems: 'flex-start',
                    '& .MuiTypography-root': { fontSize: '0.875rem' }
                  }}
                />
              </Grid>
            ))}
          </Grid>
        </FormGroup>
      </Box>
    </Card>
  );

  return (
    <Box sx={{ width: '100%' }}>
      {/* ヘッダー */}
      <Paper elevation={1} sx={{ p: 3, mb: 3, bgcolor: 'primary.50' }}>
        <Typography variant="h4" gutterBottom>
          🎯 研究室適合性評価
        </Typography>
        <Typography variant="body1" color="text.secondary">
          あなたの研究に対する希望や興味を設定して、最適な研究室を見つけましょう
        </Typography>

        {/* 設定状況の表示 */}
        <Box sx={{ mt: 2, display: 'flex', gap: 2, flexWrap: 'wrap' }}>
          <Chip
            label={`評価基準: 13項目設定済み`}
            color="primary"
            size="small"
          />
          <Chip
            label={`研究分野: ${getSelectedFieldsCount()}分野選択済み`}
            color={getSelectedFieldsCount() > 0 ? "success" : "default"}
            size="small"
          />
        </Box>
      </Paper>

      {/* エラー表示 */}
      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {/* 研究分野選択（最初に表示） */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h5" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          ✅ 興味のある研究分野を選択
          <Chip
            label={`${getSelectedFieldsCount()}分野選択中`}
            color={getSelectedFieldsCount() > 0 ? "success" : "default"}
            size="small"
          />
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          あなたが興味を持っている研究分野をチェックしてください（複数選択可）
        </Typography>

        {Object.entries(fieldCategories).map(([key, category]) =>
          renderFieldCategory(key, category)
        )}
      </Box>

      <Divider sx={{ my: 4 }} />

      {/* 評価基準設定 */}
      <Box>
        <Typography variant="h5" gutterBottom>
          📊 評価基準の設定
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          研究室選択で重視する項目を1（低）〜10（高）で設定してください
        </Typography>

        {Object.entries(criteriaCategories).map(([key, category]) =>
          renderCriteriaCategory(key, category)
        )}
      </Box>

      {/* アクション */}
      <Box sx={{ mt: 4, mb: 4 }}>
        <Paper elevation={2} sx={{ p: 3, bgcolor: 'grey.50' }}>
          <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center', mb: 2 }}>
            <Button
              variant="outlined"
              onClick={handleLoadDemo}
              disabled={isLoading}
              size="large"
            >
              📝 デモデータを読み込み
            </Button>

            <Button
              variant="contained"
              onClick={handleEvaluation}
              disabled={isLoading || getSelectedFieldsCount() === 0}
              size="large"
              sx={{ minWidth: 200 }}
            >
              {isLoading ? (
                <>
                  <CircularProgress size={20} sx={{ mr: 1 }} />
                  評価中...
                </>
              ) : (
                '🚀 研究室を評価する'
              )}
            </Button>
          </Box>

          {getSelectedFieldsCount() === 0 && (
            <Alert severity="warning" sx={{ mt: 2 }}>
              評価を実行するには、最低1つの研究分野を選択してください。
            </Alert>
          )}

          <Typography variant="body2" color="text.secondary" align="center" sx={{ mt: 2 }}>
            💡 設定完了後、「研究室を評価する」ボタンで適合性分析を開始します
          </Typography>
        </Paper>
      </Box>
    </Box>
  );
};

export default EvaluationForm;