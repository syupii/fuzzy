// frontend/src/components/ResultsList.tsx - 修正版

import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Chip,
  Grid,
  Tabs,
  Tab,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  LinearProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Paper,
  Alert
} from '@mui/material';
import {
  ExpandMore,
  StarRate,
  Science,
  School,
  TrendingUp,
  Assessment,
  Person
} from '@mui/icons-material';
import { EvaluationResponse, LabResult } from '../services/api';

interface ResultsListProps {
  data: EvaluationResponse;
}

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

const ResultsList: React.FC<ResultsListProps> = ({ data }) => {
  const [tabValue, setTabValue] = useState(0);
  const [expandedAccordion, setExpandedAccordion] = useState<string | false>(false);

  // 安全なデータアクセス
  const results = data.results || data.lab_results || [];
  const summary = data.summary || {
    total_labs: results.length,
    avg_score: 0,
    recommendations: []
  };

  // スコア色の取得
  const getScoreColor = (score: number) => {
    if (score >= 0.8) return 'success';
    if (score >= 0.6) return 'warning';
    return 'error';
  };

  // スコア値を正規化（0-1の範囲に）
  const normalizeScore = (score: number | undefined): number => {
    if (!score) return 0;
    // スコアが0-1の範囲でない場合の処理
    if (score > 1) return score / 100; // パーセンテージの場合
    return score;
  };

  // 研究室名の取得（複数のフォールバック）
  const getLabName = (labResult: LabResult, index: number): string => {
    return labResult.lab_name ||
      (labResult.lab && labResult.lab.name) ||
      `研究室 ${index + 1}`;
  };

  // 指導教員名の取得（複数のフォールバック）
  const getAdvisorName = (labResult: LabResult): string => {
    return labResult.advisor ||
      (labResult.lab && labResult.lab.professor) ||
      '指導教員情報なし';
  };

  // 総合スコアの取得（複数のフォールバック）
  const getOverallScore = (labResult: LabResult): number => {
    return normalizeScore(
      labResult.compatibility_score ||
      labResult.overall_score ||
      (labResult.compatibility && labResult.compatibility.overall_score) ||
      0
    );
  };

  // 研究室説明の取得
  const getDescription = (labResult: LabResult): string => {
    return labResult.description ||
      (labResult.lab && labResult.lab.description) ||
      labResult.specialization ||
      '詳細情報はありません';
  };

  // アコーディオン展開ハンドラー
  const handleAccordionChange = (panel: string) => (
    event: React.SyntheticEvent,
    isExpanded: boolean
  ) => {
    setExpandedAccordion(isExpanded ? panel : false);
  };

  // 研究室カード描画関数
  const renderLabCard = (labResult: LabResult, index: number) => {
    const labName = getLabName(labResult, index);
    const advisorName = getAdvisorName(labResult);
    const overallScore = getOverallScore(labResult);
    const description = getDescription(labResult);

    return (
      <Card key={index} sx={{ mb: 2 }}>
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
            <Box sx={{ flexGrow: 1 }}>
              <Typography variant="h6" component="h3" gutterBottom>
                #{index + 1} {labName}
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                <Person sx={{ mr: 1, color: 'text.secondary' }} />
                <Typography variant="subtitle1" color="text.secondary">
                  {advisorName}
                </Typography>
              </Box>

              {/* 研究分野表示 */}
              {labResult.research_fields && labResult.research_fields.length > 0 && (
                <Box sx={{ mt: 1 }}>
                  {labResult.research_fields.slice(0, 3).map((field, fieldIndex) => (
                    <Chip
                      key={fieldIndex}
                      label={field}
                      size="small"
                      variant="outlined"
                      sx={{ mr: 1, mb: 1 }}
                    />
                  ))}
                </Box>
              )}
            </Box>

            <Chip
              label={`${(overallScore * 100).toFixed(1)}%`}
              color={getScoreColor(overallScore) as any}
              sx={{
                fontSize: '1.1em',
                fontWeight: 'bold',
                height: 40,
                minWidth: 80,
                '& .MuiChip-label': {
                  fontSize: '1.1em',
                  fontWeight: 'bold'
                }
              }}
            />
          </Box>

          <Typography variant="body2" paragraph sx={{ color: 'text.secondary' }}>
            {description}
          </Typography>

          {/* 強み・考慮点の表示 */}
          <Grid container spacing={2} sx={{ mb: 2 }}>
            {labResult.strengths && labResult.strengths.length > 0 && (
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle2" color="success.main" gutterBottom>
                  強み:
                </Typography>
                <List dense>
                  {labResult.strengths.slice(0, 2).map((strength, idx) => (
                    <ListItem key={idx} sx={{ py: 0 }}>
                      <ListItemIcon sx={{ minWidth: 24 }}>
                        <StarRate color="success" fontSize="small" />
                      </ListItemIcon>
                      <ListItemText
                        primary={strength}
                        primaryTypographyProps={{ variant: 'body2' }}
                      />
                    </ListItem>
                  ))}
                </List>
              </Grid>
            )}

            {labResult.considerations && labResult.considerations.length > 0 && (
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle2" color="warning.main" gutterBottom>
                  考慮点:
                </Typography>
                <List dense>
                  {labResult.considerations.slice(0, 2).map((consideration, idx) => (
                    <ListItem key={idx} sx={{ py: 0 }}>
                      <ListItemIcon sx={{ minWidth: 24 }}>
                        <Assessment color="warning" fontSize="small" />
                      </ListItemIcon>
                      <ListItemText
                        primary={consideration}
                        primaryTypographyProps={{ variant: 'body2' }}
                      />
                    </ListItem>
                  ))}
                </List>
              </Grid>
            )}
          </Grid>

          {/* 詳細スコア（アコーディオン形式） */}
          {labResult.compatibility && labResult.compatibility.criterion_scores && (
            <Accordion
              expanded={expandedAccordion === `lab-${index}`}
              onChange={handleAccordionChange(`lab-${index}`)}
            >
              <AccordionSummary expandIcon={<ExpandMore />}>
                <Typography variant="subtitle2">詳細評価スコア</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Grid container spacing={2}>
                  {Object.entries(labResult.compatibility.criterion_scores).map(([criterion, scoreData]: [string, any], scoreIndex: number) => {
                    const score = normalizeScore(scoreData.score || scoreData.similarity || scoreData);
                    const displayName = criterion.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());

                    return (
                      <Grid item xs={12} sm={6} md={4} key={scoreIndex}>
                        <Box sx={{ mb: 1 }}>
                          <Typography variant="caption" display="block" gutterBottom>
                            {displayName}: {(score * 100).toFixed(0)}%
                          </Typography>
                          <LinearProgress
                            variant="determinate"
                            value={score * 100}
                            color={getScoreColor(score) as any}
                            sx={{ height: 6, borderRadius: 3 }}
                          />
                        </Box>
                      </Grid>
                    );
                  })}
                </Grid>
              </AccordionDetails>
            </Accordion>
          )}
        </CardContent>
      </Card>
    );
  };

  // 結果が空の場合
  if (!results || results.length === 0) {
    return (
      <Paper sx={{ p: 4, textAlign: 'center' }}>
        <Assessment sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
        <Typography variant="h6" gutterBottom>
          評価結果がありません
        </Typography>
        <Typography variant="body2" color="text.secondary">
          評価を実行して研究室のマッチング結果を確認してください
        </Typography>
      </Paper>
    );
  }

  return (
    <Box sx={{ width: '100%' }}>
      <Tabs
        value={tabValue}
        onChange={(_, newValue) => setTabValue(newValue)}
        variant="fullWidth"
        sx={{ borderBottom: 1, borderColor: 'divider' }}
      >
        <Tab
          label={`研究室一覧 (${results.length})`}
          icon={<School />}
          iconPosition="start"
        />
        <Tab
          label="評価サマリー"
          icon={<Assessment />}
          iconPosition="start"
        />
      </Tabs>

      <TabPanel value={tabValue} index={0}>
        <Box sx={{ mb: 3 }}>
          <Typography variant="h5" gutterBottom>
            研究室マッチング結果
          </Typography>
          <Typography variant="body2" color="text.secondary">
            適合度の高い順に表示しています（{results.length}件の研究室）
          </Typography>
        </Box>

        {results.map((lab: LabResult, index: number) => renderLabCard(lab, index))}
      </TabPanel>

      <TabPanel value={tabValue} index={1}>
        <Typography variant="h5" gutterBottom>
          評価サマリー
        </Typography>

        <Grid container spacing={3} sx={{ mb: 4 }}>
          <Grid item xs={12} md={4}>
            <Card>
              <CardContent sx={{ textAlign: 'center' }}>
                <School sx={{ fontSize: 48, color: 'primary.main', mb: 1 }} />
                <Typography variant="h4" color="primary">
                  {summary.total_labs || results.length}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  評価対象研究室数
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={4}>
            <Card>
              <CardContent sx={{ textAlign: 'center' }}>
                <TrendingUp sx={{ fontSize: 48, color: 'success.main', mb: 1 }} />
                <Typography variant="h4" color="success.main">
                  {((summary.avg_score || 0) * 100).toFixed(1)}%
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  平均適合度
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={4}>
            <Card>
              <CardContent sx={{ textAlign: 'center' }}>
                <StarRate sx={{ fontSize: 48, color: 'warning.main', mb: 1 }} />
                <Typography variant="h4" color="warning.main">
                  {summary.best_match_lab || (results[0] ? getLabName(results[0], 0) : '−')}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  最高適合研究室
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>

        {/* 推奨事項 */}
        {summary.recommendations && summary.recommendations.length > 0 && (
          <Box>
            <Typography variant="h6" gutterBottom>
              推奨事項
            </Typography>
            <Card>
              <CardContent>
                <List>
                  {summary.recommendations.map((rec: string, index: number) => (
                    <ListItem key={index}>
                      <ListItemIcon>
                        <StarRate color="primary" />
                      </ListItemIcon>
                      <ListItemText primary={rec} />
                    </ListItem>
                  ))}
                </List>
              </CardContent>
            </Card>
          </Box>
        )}

        {/* 分野分析（もし利用可能な場合） */}
        {summary.field_analysis && (
          <Box sx={{ mt: 4 }}>
            <Typography variant="h6" gutterBottom>
              分野別分析
            </Typography>
            <Card>
              <CardContent>
                <Typography variant="body2">
                  {JSON.stringify(summary.field_analysis, null, 2)}
                </Typography>
              </CardContent>
            </Card>
          </Box>
        )}
      </TabPanel>
    </Box>
  );
};

export default ResultsList;