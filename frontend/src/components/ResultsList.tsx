// ResultsList.tsx - 修正版
import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  LinearProgress,
  Chip,
  Grid,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Avatar,
  Paper,
  Divider,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Tooltip,
  Tabs,
  Tab,
} from '@mui/material';
import {
  ExpandMore,
  School,
  Person,
  Psychology,
  EmojiEvents,
  TrendingUp,
  Science,
  Category,
  StarRate,
  Assessment,
} from '@mui/icons-material';
import { EvaluationResponse, fieldUtils } from '../services/api';

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
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

const ResultsList: React.FC<ResultsListProps> = ({ data }) => {
  const { results, summary } = data;
  const [tabValue, setTabValue] = useState(0);
  const hasFieldAnalysis = summary.field_analysis && summary.field_analysis.selected_fields_count > 0;

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const getScoreColor = (score: number): 'success' | 'warning' | 'error' => {
    if (score >= 80) return 'success';
    if (score >= 60) return 'warning';
    return 'error';
  };

  const getScoreIcon = (score: number) => {
    if (score >= 85) return '🎯';
    if (score >= 70) return '✅';
    if (score >= 50) return '👍';
    return '⚠️';
  };

  const getRankIcon = (rank: number) => {
    if (rank === 1) return '🥇';
    if (rank === 2) return '🥈';
    if (rank === 3) return '🥉';
    return `${rank}位`;
  };

  const criteriaLabels: Record<string, string> = {
    research_intensity: '研究強度',
    advisor_style: '指導スタイル',
    team_work: 'チームワーク',
    workload: 'ワークロード',
    theory_practice: '理論・実践',
    research_field_match: '分野適合性',
    skill_development: 'スキル開発',
    learning_pace: '学習ペース',
    difficulty_preference: '難易度選好',
    communication_style: 'コミュニケーション',
    meeting_frequency: 'ミーティング頻度',
    lab_atmosphere: '研究室雰囲気',
    innovation_risk: '革新性・リスク',
    methodology_preference: '手法選好',
    interdisciplinary: '学際性',
    flexibility: '柔軟性',
    evening_weekend_work: '夜間・休日作業',
    publication_opportunity: '論文機会',
    financial_support: '経済支援',
    lab_hierarchy: '研究室階層',
    core_time_flexibility: 'コアタイム柔軟性'
  };

  const criteriaEmojis: Record<string, string> = {
    research_intensity: '🔬',
    advisor_style: '👨‍🏫',
    team_work: '🤝',
    workload: '⚡',
    theory_practice: '⚖️',
    research_field_match: '🎯',
    skill_development: '📈',
    learning_pace: '🏃',
    difficulty_preference: '🎢',
    communication_style: '💬',
    meeting_frequency: '📅',
    lab_atmosphere: '🌟',
    innovation_risk: '🚀',
    methodology_preference: '🔧',
    interdisciplinary: '🌐',
    flexibility: '🤸',
    evening_weekend_work: '🌙',
    publication_opportunity: '📝',
    financial_support: '💰',
    lab_hierarchy: '👥',
    core_time_flexibility: '⏰'
  };

  return (
    <Box sx={{ mt: 3 }}>
      {/* 拡張サマリーカード */}
      <Paper elevation={3} sx={{ mb: 4, background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', color: 'white' }}>
        <Box sx={{ p: 3 }}>
          <Box sx={{ textAlign: 'center', mb: 3 }}>
            <EmojiEvents sx={{ fontSize: 48, mb: 1 }} />
            <Typography variant="h5" gutterBottom>
              📊 評価サマリー
            </Typography>
          </Box>

          <Grid container spacing={3}>
            <Grid item xs={12} sm={3}>
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="h3" fontWeight="bold">
                  {summary.total_labs}
                </Typography>
                <Typography variant="body2" sx={{ opacity: 0.9 }}>
                  評価対象研究室
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} sm={3}>
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="h3" fontWeight="bold">
                  {summary.avg_score.toFixed(1)}
                </Typography>
                <Typography variant="body2" sx={{ opacity: 0.9 }}>
                  平均適合度
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} sm={3}>
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="h3" fontWeight="bold">
                  {results.length > 0 ? results[0].compatibility.overall_score.toFixed(1) : '0'}
                </Typography>
                <Typography variant="body2" sx={{ opacity: 0.9 }}>
                  最高適合度
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} sm={3}>
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="h3" fontWeight="bold">
                  {hasFieldAnalysis ? (summary.field_analysis!.selected_fields_count || 0) : '0'}
                </Typography>
                <Typography variant="body2" sx={{ opacity: 0.9 }}>
                  選択分野数
                </Typography>
              </Box>
            </Grid>
          </Grid>
        </Box>
      </Paper>

      {/* タブナビゲーション */}
      <Paper sx={{ mb: 3 }}>
        <Tabs value={tabValue} onChange={handleTabChange} aria-label="結果表示タブ">
          <Tab label="研究室ランキング" />
          <Tab label="評価分析" />
          <Tab label="分野別比較" />
        </Tabs>
      </Paper>

      {/* タブ1: 研究室ランキング */}
      <TabPanel value={tabValue} index={0}>
        <Grid container spacing={3}>
          {results.map((result, index) => (
            <Grid item xs={12} key={result.lab.id}>
              <Card
                elevation={result.ranking_position <= 3 ? 8 : 2}
                sx={{
                  position: 'relative',
                  border: result.ranking_position <= 3 ? '2px solid gold' : 'none',
                  background: result.ranking_position === 1 ?
                    'linear-gradient(135deg, #fff9c4 0%, #fffacd 100%)' :
                    'background.paper',
                }}
              >
                <CardContent>
                  {/* ヘッダー部分 */}
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                      <Avatar sx={{
                        bgcolor: getScoreColor(result.compatibility.overall_score) === 'success' ? 'success.main' :
                          getScoreColor(result.compatibility.overall_score) === 'warning' ? 'warning.main' : 'error.main'
                      }}>
                        <School />
                      </Avatar>
                      <Box>
                        <Typography variant="h6" fontWeight="bold">
                          {getRankIcon(index + 1)} {result.lab.name}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          {result.lab.professor} 教授 | {result.lab.research_area}
                        </Typography>
                      </Box>
                    </Box>

                    {/* スコア表示 */}
                    <Box sx={{ textAlign: 'right' }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Typography variant="h4" fontWeight="bold"
                          color={getScoreColor(result.compatibility.overall_score) + '.main'}>
                          {result.compatibility.overall_score.toFixed(1)}
                        </Typography>
                        <Typography variant="h6">
                          {getScoreIcon(result.compatibility.overall_score)}
                        </Typography>
                      </Box>
                      <Typography variant="body2" color="text.secondary">
                        適合度スコア
                      </Typography>
                    </Box>
                  </Box>

                  {/* プログレスバー */}
                  <LinearProgress
                    variant="determinate"
                    value={result.compatibility.overall_score}
                    color={getScoreColor(result.compatibility.overall_score)}
                    sx={{ height: 8, borderRadius: 4, mb: 3 }}
                  />

                  {/* 研究室詳細 */}
                  <Accordion>
                    <AccordionSummary expandIcon={<ExpandMore />}>
                      <Typography variant="h6">📊 詳細分析</Typography>
                    </AccordionSummary>
                    <AccordionDetails>
                      <Grid container spacing={2}>
                        <Grid item xs={12} md={6}>
                          <Typography variant="h6" gutterBottom>
                            📊 適合度スコア
                          </Typography>
                          <List dense>
                            {Object.entries(result.compatibility.criterion_scores).slice(0, 8).map(([criterion, score]) => (
                              <ListItem key={criterion}>
                                <ListItemIcon>
                                  <Typography>
                                    {criteriaEmojis[criterion as keyof typeof criteriaEmojis] || '📊'}
                                  </Typography>
                                </ListItemIcon>
                                <ListItemText
                                  primary={criteriaLabels[criterion as keyof typeof criteriaLabels] || criterion}
                                  secondary={
                                    <Box sx={{ mt: 1 }}>
                                      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                                        <Typography variant="body2">
                                          類似度: {(score.similarity * 100).toFixed(0)}%
                                        </Typography>
                                        <Typography variant="body2">
                                          重み: {score.weight.toFixed(2)}
                                        </Typography>
                                      </Box>
                                      <LinearProgress
                                        variant="determinate"
                                        value={score.similarity * 100}
                                        color={score.similarity >= 0.7 ? 'success' : score.similarity >= 0.5 ? 'warning' : 'error'}
                                        sx={{ height: 4 }}
                                      />
                                    </Box>
                                  }
                                />
                              </ListItem>
                            ))}
                          </List>
                        </Grid>

                        <Grid item xs={12} md={6}>
                          <Typography variant="h6" gutterBottom>
                            🏫 研究室情報
                          </Typography>
                          <List dense>
                            <ListItem>
                              <ListItemIcon><Person /></ListItemIcon>
                              <ListItemText
                                primary="教授"
                                secondary={result.lab.professor}
                              />
                            </ListItem>
                            <ListItem>
                              <ListItemIcon><Science /></ListItemIcon>
                              <ListItemText
                                primary="研究分野"
                                secondary={result.lab.research_area}
                              />
                            </ListItem>
                            <ListItem>
                              <ListItemIcon><Category /></ListItemIcon>
                              <ListItemText
                                primary="専門領域"
                                secondary={result.lab.specialization}
                              />
                            </ListItem>
                          </List>
                        </Grid>
                      </Grid>
                    </AccordionDetails>
                  </Accordion>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </TabPanel>

      {/* タブ2: 評価分析 */}
      <TabPanel value={tabValue} index={1}>
        <Box>
          <Typography variant="h5" gutterBottom>
            評価分析
          </Typography>
          <Typography>
            詳細な評価分析内容をここに追加
          </Typography>
        </Box>
      </TabPanel>

      {/* タブ3: 分野別比較 */}
      <TabPanel value={tabValue} index={2}>
        <Box>
          <Typography variant="h5" gutterBottom>
            分野別比較
          </Typography>
          <Typography>
            分野別比較内容をここに追加
          </Typography>
        </Box>
      </TabPanel>
    </Box>
  );
};

export default ResultsList;