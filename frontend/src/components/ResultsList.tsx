// src/components/ResultsList.tsx - 完全修正版
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
  Groups,
  Explore,
  Schedule,
  Article,
  AccountTree,
  ConnectWithoutContact,
  Lightbulb,
  AccessTime,
  Settings
} from '@mui/icons-material';
import { EvaluationResponse, CRITERIA_INFO, fieldUtils } from '../services/api';

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

  // 13項目の評価基準ラベル
  const criteriaLabels: Record<string, string> = {
    // 基本項目（5項目）
    research_intensity: "研究強度",
    advisor_style: "指導スタイル",
    team_work: "チームワーク",
    workload: "ワークロード",
    theory_practice: "理論・実践バランス",

    // 拡張項目（5項目）
    research_field_match: "研究分野適合性",
    skill_development: "スキル開発",
    lab_atmosphere: "研究室雰囲気",
    flexibility: "柔軟性",
    publication_opportunity: "論文発表機会",

    // 特殊項目（3項目）
    interdisciplinary: "学際性",
    communication_style: "コミュニケーション",
    innovation_risk: "革新性・リスク許容度"
  };

  const getCriteriaIcon = (criteriaKey: string) => {
    const iconMap: { [key: string]: React.ReactElement } = {
      research_intensity: <Science fontSize="small" />,
      advisor_style: <School fontSize="small" />,
      team_work: <Groups fontSize="small" />,
      workload: <Schedule fontSize="small" />,
      theory_practice: <Psychology fontSize="small" />,
      research_field_match: <Explore fontSize="small" />,
      skill_development: <TrendingUp fontSize="small" />,
      lab_atmosphere: <Groups fontSize="small" />,
      flexibility: <AccessTime fontSize="small" />,
      publication_opportunity: <Article fontSize="small" />,
      interdisciplinary: <AccountTree fontSize="small" />,
      communication_style: <ConnectWithoutContact fontSize="small" />,
      innovation_risk: <Lightbulb fontSize="small" />
    };
    return iconMap[criteriaKey] || <Settings fontSize="small" />;
  };

  const renderLabCard = (lab: any, index: number) => {
    return (
      <Card key={lab.lab_id} sx={{ mb: 3, border: index < 3 ? 2 : 1, borderColor: index < 3 ? 'primary.main' : 'divider' }}>
        <CardContent>
          {/* ヘッダー部分 */}
          <Box display="flex" justifyContent="space-between" alignItems="flex-start" mb={2}>
            <Box flex={1}>
              <Box display="flex" alignItems="center" gap={1} mb={1}>
                <Avatar sx={{ bgcolor: index < 3 ? 'primary.main' : 'grey.400' }}>
                  {getRankIcon(lab.rank)}
                </Avatar>
                <Box>
                  <Typography variant="h6" component="h2">
                    {lab.lab_name}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    指導教員: {lab.advisor}
                  </Typography>
                </Box>
              </Box>
            </Box>

            <Box textAlign="right">
              <Box display="flex" alignItems="center" gap={1} mb={1}>
                <Typography variant="h4" color="primary.main" fontWeight="bold">
                  {lab.overall_score.toFixed(1)}
                </Typography>
                <Typography variant="body2" color="text.secondary">点</Typography>
                <Chip
                  label={getScoreIcon(lab.overall_score)}
                  size="small"
                  color={getScoreColor(lab.overall_score)}
                />
              </Box>
              <LinearProgress
                variant="determinate"
                value={lab.overall_score}
                color={getScoreColor(lab.overall_score)}
                sx={{ width: 100, height: 8, borderRadius: 4 }}
              />
            </Box>
          </Box>

          {/* 基本情報 */}
          <Typography variant="body2" paragraph>
            {lab.description}
          </Typography>

          {/* 研究分野 */}
          <Box mb={2}>
            <Typography variant="subtitle2" gutterBottom>
              🔬 研究分野
            </Typography>
            <Box display="flex" gap={1} flexWrap="wrap">
              {lab.research_areas?.map((area: string, idx: number) => (
                <Chip
                  key={idx}
                  label={area}
                  size="small"
                  variant="outlined"
                  color="secondary"
                />
              ))}
            </Box>
          </Box>

          {/* 詳細スコア */}
          <Accordion>
            <AccordionSummary expandIcon={<ExpandMore />}>
              <Typography variant="subtitle2">
                📊 詳細評価スコア（13項目）
              </Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={2}>
                {Object.entries(lab.detailed_scores || {}).map(([criteria, scoreValue]) => {
                  const label = criteriaLabels[criteria] || criteria;
                  // 型安全な数値変換
                  const score = typeof scoreValue === 'number' ? scoreValue : Number(scoreValue) || 0;

                  return (
                    <Grid item xs={12} sm={6} md={4} key={criteria}>
                      <Box display="flex" alignItems="center" gap={1} mb={1}>
                        {getCriteriaIcon(criteria)}
                        <Typography variant="body2" fontSize="0.8rem">
                          {label}
                        </Typography>
                      </Box>
                      <Box display="flex" alignItems="center" gap={1}>
                        <LinearProgress
                          variant="determinate"
                          value={score}
                          sx={{ flex: 1, height: 6 }}
                          color={getScoreColor(score)}
                        />
                        <Typography variant="body2" fontWeight="bold" minWidth="40px">
                          {score.toFixed(1)}
                        </Typography>
                      </Box>
                    </Grid>
                  );
                })}
              </Grid>
            </AccordionDetails>
          </Accordion>

          {/* 分野適合性と追加情報 */}
          <Box mt={2}>
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <Box>
                  <Typography variant="subtitle2" gutterBottom>
                    🎯 分野適合性
                  </Typography>
                  <Box display="flex" alignItems="center" gap={1}>
                    <LinearProgress
                      variant="determinate"
                      value={lab.field_compatibility}
                      sx={{ flex: 1 }}
                      color={getScoreColor(lab.field_compatibility)}
                    />
                    <Typography variant="body2" fontWeight="bold">
                      {lab.field_compatibility.toFixed(1)}%
                    </Typography>
                  </Box>
                </Box>
              </Grid>

              {lab.publications && (
                <Grid item xs={12} md={6}>
                  <Typography variant="subtitle2" gutterBottom>
                    📄 論文実績
                  </Typography>
                  <Typography variant="body2">
                    年間約 {lab.publications} 本
                  </Typography>
                </Grid>
              )}
            </Grid>
          </Box>

          {/* 強みと注意点 */}
          <Box mt={2}>
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle2" gutterBottom color="success.main">
                  ✅ 強み・メリット
                </Typography>
                <List dense>
                  {lab.strengths?.map((strength: string, idx: number) => (
                    <ListItem key={idx} sx={{ py: 0.5 }}>
                      <ListItemIcon sx={{ minWidth: 20 }}>
                        <Typography variant="body2">•</Typography>
                      </ListItemIcon>
                      <ListItemText primary={strength} />
                    </ListItem>
                  ))}
                </List>
              </Grid>

              <Grid item xs={12} md={6}>
                <Typography variant="subtitle2" gutterBottom color="warning.main">
                  ⚠️ 検討事項
                </Typography>
                <List dense>
                  {lab.considerations?.map((consideration: string, idx: number) => (
                    <ListItem key={idx} sx={{ py: 0.5 }}>
                      <ListItemIcon sx={{ minWidth: 20 }}>
                        <Typography variant="body2">•</Typography>
                      </ListItemIcon>
                      <ListItemText primary={consideration} />
                    </ListItem>
                  ))}
                </List>
              </Grid>
            </Grid>
          </Box>
        </CardContent>
      </Card>
    );
  };

  const renderSummary = () => {
    return (
      <Grid container spacing={3}>
        {/* 基本統計 */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Assessment color="primary" sx={{ fontSize: 40, mb: 1 }} />
              <Typography variant="h4" color="primary">
                {summary.total_labs_evaluated}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                評価対象研究室
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <TrendingUp color="success" sx={{ fontSize: 40, mb: 1 }} />
              <Typography variant="h4" color="success.main">
                {summary.avg_score.toFixed(1)}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                平均適合スコア
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <EmojiEvents color="warning" sx={{ fontSize: 40, mb: 1 }} />
              <Typography variant="h4" color="warning.main">
                {summary.top_score.toFixed(1)}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                最高スコア
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* 評価基準分析 */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                📊 評価基準の重要度分析
              </Typography>
              <Grid container spacing={2}>
                {Object.entries(summary.criteria_analysis || {}).map(([criteria, analysis]) => {
                  const label = criteriaLabels[criteria] || criteria;
                  const weight = typeof analysis === 'object' && analysis !== null && 'weight' in analysis
                    ? Number((analysis as any).weight) : 0;

                  return (
                    <Grid item xs={12} sm={6} md={4} key={criteria}>
                      <Box>
                        <Box display="flex" alignItems="center" gap={1} mb={1}>
                          {getCriteriaIcon(criteria)}
                          <Typography variant="body2">
                            {label}
                          </Typography>
                        </Box>
                        <Box display="flex" alignItems="center" gap={1}>
                          <Typography variant="caption" color="text.secondary">
                            重要度:
                          </Typography>
                          <LinearProgress
                            variant="determinate"
                            value={weight * 100}
                            sx={{ flex: 1, height: 4 }}
                          />
                          <Typography variant="caption">
                            {(weight * 100).toFixed(0)}%
                          </Typography>
                        </Box>
                      </Box>
                    </Grid>
                  );
                })}
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* 分野分析 */}
        {hasFieldAnalysis && (
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  🔬 研究分野の興味分析
                </Typography>
                <Typography variant="body2" color="text.secondary" paragraph>
                  選択された分野数: {summary.field_analysis.selected_fields_count}
                </Typography>

                <Box mb={2}>
                  <Typography variant="subtitle2" gutterBottom>
                    主要な興味分野:
                  </Typography>
                  <Box display="flex" gap={1} flexWrap="wrap">
                    {summary.field_analysis.primary_interests?.map((field, idx) => (
                      <Chip
                        key={idx}
                        label={field}
                        color="primary"
                        variant="outlined"
                      />
                    ))}
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        )}

        {/* 推奨事項 */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                💡 推奨事項
              </Typography>
              <List>
                {summary.recommendations?.map((recommendation, idx) => (
                  <ListItem key={idx}>
                    <ListItemIcon>
                      <StarRate color="primary" />
                    </ListItemIcon>
                    <ListItemText primary={recommendation} />
                  </ListItem>
                ))}
              </List>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    );
  };

  return (
    <Box>
      <Paper sx={{ mb: 3 }}>
        <Tabs value={tabValue} onChange={handleTabChange} sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tab icon={<School />} label="研究室ランキング" />
          <Tab icon={<Assessment />} label="分析サマリー" />
        </Tabs>
      </Paper>

      <TabPanel value={tabValue} index={0}>
        <Typography variant="h5" gutterBottom>
          🏆 研究室適合ランキング
        </Typography>
        <Typography variant="body2" color="text.secondary" paragraph>
          あなたの評価基準（13項目）と研究分野の興味に基づいて算出された適合度ランキングです
        </Typography>

        {results.map((lab, index) => renderLabCard(lab, index))}
      </TabPanel>

      <TabPanel value={tabValue} index={1}>
        <Typography variant="h5" gutterBottom>
          📈 詳細分析レポート
        </Typography>
        <Typography variant="body2" color="text.secondary" paragraph>
          評価結果の統計情報と分析データです
        </Typography>

        {renderSummary()}
      </TabPanel>
    </Box>
  );
};

export default ResultsList;