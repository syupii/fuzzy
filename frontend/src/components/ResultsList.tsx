// frontend/src/components/ResultsList.tsx - シンプル版
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
import { EvaluationResponse } from '../services/api';

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

  // アコーディオン展開ハンドラー
  const handleAccordionChange = (panel: string) => (
    event: React.SyntheticEvent,
    isExpanded: boolean
  ) => {
    setExpandedAccordion(isExpanded ? panel : false);
  };

  // 研究室カード描画関数
  const renderLabCard = (labResult: any, index: number) => {
    const lab = labResult.lab || {};
    const labName = labResult.lab_name || lab.name || `研究室 ${index + 1}`;
    const professor = labResult.advisor || lab.professor || '指導教員情報なし';
    const overallScore = labResult.overall_score || (labResult.compatibility ? labResult.compatibility.overall_score : 0) || 0;
    const description = lab.description || labResult.description || '詳細情報はありません';

    return (
      <Card key={index} sx={{ mb: 2 }}>
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
            <Box>
              <Typography variant="h6" component="h3">
                #{index + 1} {labName}
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                <Person sx={{ mr: 1, color: 'text.secondary' }} />
                <Typography variant="subtitle1" color="text.secondary">
                  {professor}
                </Typography>
              </Box>
            </Box>
            <Chip
              label={`${(overallScore * 100).toFixed(1)}%`}
              color={getScoreColor(overallScore) as any}
              sx={{
                fontSize: '1.1em',
                fontWeight: 'bold',
                height: 40,
                '& .MuiChip-label': {
                  fontSize: '1.1em',
                  fontWeight: 'bold'
                }
              }}
            />
          </Box>

          <Typography variant="body2" paragraph>
            {description}
          </Typography>

          {/* 詳細スコア */}
          {labResult.compatibility && labResult.compatibility.criterion_scores && (
            <Accordion
              expanded={expandedAccordion === `lab-${index}`}
              onChange={handleAccordionChange(`lab-${index}`)}
              sx={{ mt: 2 }}
            >
              <AccordionSummary expandIcon={<ExpandMore />}>
                <Typography variant="subtitle2">詳細評価スコア</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Grid container spacing={1}>
                  {Object.entries(labResult.compatibility.criterion_scores).map(([criterion, scoreData]: [string, any], scoreIndex: number) => {
                    const score = scoreData.score || scoreData.similarity || 0;

                    return (
                      <Grid item xs={12} sm={6} key={scoreIndex}>
                        <Box sx={{ mb: 1 }}>
                          <Typography variant="caption" display="block">
                            {criterion}: {(score * 100).toFixed(0)}%
                          </Typography>
                          <LinearProgress
                            variant="determinate"
                            value={score * 100}
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

          {/* 推奨事項 */}
          {labResult.recommendations && labResult.recommendations.length > 0 && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                推奨ポイント:
              </Typography>
              <List dense>
                {labResult.recommendations.map((rec: string, recIndex: number) => (
                  <ListItem key={recIndex}>
                    <ListItemIcon>
                      <StarRate color="primary" fontSize="small" />
                    </ListItemIcon>
                    <ListItemText primary={rec} />
                  </ListItem>
                ))}
              </List>
            </Box>
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
        <Typography variant="h5" gutterBottom>
          研究室マッチング結果
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          適合度の高い順に表示しています
        </Typography>

        {results.map((lab: any, index: number) => renderLabCard(lab, index))}
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
                  {summary.best_match_lab || (results[0] && (results[0].lab_name || results[0].lab?.name)) || '−'}
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
          <Box sx={{ mt: 4 }}>
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
      </TabPanel>
    </Box>
  );
};

export default ResultsList;