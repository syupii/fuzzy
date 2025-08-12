import React from 'react';
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
} from '@mui/material';
import {
  ExpandMore,
  School,
  Person,
  Psychology,
  EmojiEvents,
} from '@mui/icons-material';
import { EvaluationResponse } from '../services/api';

interface ResultsListProps {
  data: EvaluationResponse;
}

const ResultsList: React.FC<ResultsListProps> = ({ data }) => {
  const { results, summary } = data;

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
    if (rank === 1) return '1';
    if (rank === 2) return '2';
    if (rank === 3) return '3';
    return `${rank}位`;
  };

  const criteriaLabels = {
    research_intensity: '研究強度',
    advisor_style: '指導スタイル',
    team_work: 'チームワーク',
    workload: 'ワークロード',
    theory_practice: '理論・実践'
  };

  const criteriaEmojis = {
    research_intensity: '🔬',
    advisor_style: '👨‍🏫',
    team_work: '🤝',
    workload: '⚡',
    theory_practice: '⚖️'
  };

  return (
    <Box>
      {/* サマリーカード */}
      <Paper elevation={2} sx={{ mb: 4, background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' }}>
        <CardContent sx={{ color: 'white' }}>
          <Box sx={{ textAlign: 'center', mb: 3 }}>
            <EmojiEvents sx={{ fontSize: 48, mb: 1 }} />
            <Typography variant="h5" gutterBottom>
              📊 評価サマリー
            </Typography>
          </Box>
          
          <Grid container spacing={3}>
            <Grid item xs={12} sm={4}> {/* ここに 'item' を追加 */}
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="h3" fontWeight="bold">
                  {summary.total_labs}
                </Typography>
                <Typography variant="body2" sx={{ opacity: 0.9 }}>
                  評価対象研究室
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} sm={4}> {/* ここに 'item' を追加 */}
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="h3" fontWeight="bold">
                  {summary.avg_score}
                </Typography>
                <Typography variant="body2" sx={{ opacity: 0.9 }}>
                  平均適合度
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} sm={4}> {/* ここに 'item' を追加 */}
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="h6" gutterBottom>
                  🏆 最高適合研究室
                </Typography>
                <Typography variant="h6" fontWeight="bold">
                  {summary.best_match}
                </Typography>
              </Box>
            </Grid>
          </Grid>
        </CardContent>
      </Paper>

      {/* 結果一覧 */}
      <Typography variant="h5" gutterBottom sx={{ mt: 4, mb: 3 }}>
        🏆 適合度ランキング
      </Typography>

      {results.map((result, index) => (
        <Card key={result.lab.id} sx={{ mb: 3 }} elevation={2}>
          <CardContent>
            {/* ヘッダー */}
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
              <Avatar 
                sx={{ 
                  bgcolor: index < 3 ? 'gold' : 'primary.main', 
                  mr: 2,
                  width: 56,
                  height: 56,
                  fontSize: '1.2rem'
                }}
              >
                {getRankIcon(index + 1)}
              </Avatar>
              <Box sx={{ flexGrow: 1 }}>
                <Typography variant="h6" gutterBottom>
                  {result.lab.name}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  <Person sx={{ fontSize: 16, mr: 0.5 }} />
                  {result.lab.professor} | {result.lab.research_area}
                </Typography>
              </Box>
              <Box sx={{ textAlign: 'right' }}>
                <Typography 
                  variant="h3" 
                  color={`${getScoreColor(result.compatibility.overall_score)}.main`}
                  fontWeight="bold"
                >
                  {getScoreIcon(result.compatibility.overall_score)} {result.compatibility.overall_score}
                </Typography>
                <Chip 
                  label={`信頼度: ${result.compatibility.confidence}%`}
                  size="small"
                  color={getScoreColor(result.compatibility.confidence)}
                  sx={{ mt: 1 }}
                />
              </Box>
            </Box>

            {/* 全体適合度バー */}
            <Box sx={{ mb: 3 }}>
              <Typography variant="body2" gutterBottom fontWeight="medium">
                総合適合度
              </Typography>
              <LinearProgress
                variant="determinate"
                value={result.compatibility.overall_score}
                color={getScoreColor(result.compatibility.overall_score)}
                sx={{ height: 12, borderRadius: 6 }}
              />
            </Box>

            {/* 詳細分析アコーディオン */}
            <Accordion>
              <AccordionSummary expandIcon={<ExpandMore />}>
                <Typography sx={{ display: 'flex', alignItems: 'center' }}>
                  <Psychology sx={{ mr: 1 }} />
                  詳細分析を見る
                </Typography>
              </AccordionSummary>
              <AccordionDetails>
                {/* 各基準のスコア */}
                <Typography variant="h6" gutterBottom sx={{ mt: 2, mb: 2 }}>
                  📊 基準別適合度
                </Typography>
                <Grid container spacing={2} sx={{ mb: 3 }}>
                  {Object.entries(result.compatibility.criterion_scores).map(([criterion, scoreData]) => (
                    <Grid item xs={12} sm={6} md={4} key={criterion as string}> {/* ここにも 'item' を追加済みなので確認 */}
                      <Paper elevation={1} sx={{ p: 2 }}>
                        <Typography variant="body2" gutterBottom fontWeight="medium">
                          {criteriaEmojis[criterion as keyof typeof criteriaEmojis]} {criteriaLabels[criterion as keyof typeof criteriaLabels]}
                        </Typography>
                        <LinearProgress
                          variant="determinate"
                          value={scoreData.similarity * 100}
                          color={getScoreColor(scoreData.similarity * 100)}
                          sx={{ height: 8, borderRadius: 4, mb: 1 }}
                        />
                        <Typography variant="caption" color="text.secondary">
                          あなた: {scoreData.user_preference} | 研究室: {scoreData.lab_feature}
                        </Typography>
                      </Paper>
                    </Grid>
                  ))}
                </Grid>

                {/* 説明文 */}
                <Box sx={{ bgcolor: 'grey.50', p: 3, borderRadius: 2 }}>
                  <Typography variant="h6" gutterBottom>
                    💡 AI分析結果
                  </Typography>
                  <Typography variant="body2" sx={{ lineHeight: 1.6 }}>
                    {result.compatibility.explanation}
                  </Typography>
                </Box>

                {/* 研究室情報 */}
                <Box sx={{ mt: 3, p: 3, bgcolor: 'blue.50', borderRadius: 2 }}>
                  <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                    <School sx={{ mr: 1 }} />
                    研究室概要
                  </Typography>
                  <Typography variant="body2" sx={{ lineHeight: 1.6 }}>
                    {result.lab.description}
                  </Typography>
                </Box>
              </AccordionDetails>
            </Accordion>
          </CardContent>
        </Card>
      ))}
    </Box>
  );
};

export default ResultsList;