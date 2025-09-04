// src/components/ResultsList.tsx - 簡易版
import React from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  Chip,
  LinearProgress,
} from '@mui/material';
import { Science, Person, TrendingUp } from '@mui/icons-material';
import { EvaluationResponse } from '../services/api';

interface ResultsListProps {
  data: EvaluationResponse;
}

const ResultsList: React.FC<ResultsListProps> = ({ data }) => {
  const getScoreColor = (score: number): 'success' | 'warning' | 'error' => {
    if (score >= 70) return 'success';
    if (score >= 50) return 'warning';
    return 'error';
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom sx={{ textAlign: 'center', mb: 4 }}>
        <Science sx={{ mr: 2, verticalAlign: 'middle' }} />
        研究室適合度評価結果
      </Typography>

      {/* システム情報 */}
      <Card sx={{ mb: 3, backgroundColor: 'primary.light', color: 'primary.contrastText' }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>評価サマリー</Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={4}>
              <Typography variant="body2">評価対象: {data.total_labs_evaluated}研究室</Typography>
            </Grid>
            <Grid item xs={12} sm={4}>
              <Typography variant="body2">
                ファジィ推論: {data.system_info.fuzzy_enabled ? '✅' : '❌'}
              </Typography>
            </Grid>
            <Grid item xs={12} sm={4}>
              <Typography variant="body2">
                遺伝的アルゴリズム: {data.system_info.genetic_enabled ? '✅' : '❌'}
              </Typography>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* 研究室結果 */}
      <Box>
        {data.evaluation_results.map((result, index) => (
          <Card key={result.lab_id} sx={{ mb: 3 }}>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Box>
                  <Typography variant="h5" component="h3">
                    {result.lab_name}
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ display: 'flex', alignItems: 'center' }}>
                    <Person sx={{ mr: 0.5, fontSize: 16 }} />
                    {result.advisor}
                  </Typography>
                </Box>
                <Chip
                  label={`第${index + 1}位`}
                  color={index === 0 ? 'primary' : 'default'}
                  variant={index === 0 ? 'filled' : 'outlined'}
                  size="medium"
                />
              </Box>

              {/* 総合適合度 */}
              <Box sx={{ mb: 3 }}>
                <Typography variant="h6" gutterBottom>
                  <TrendingUp sx={{ mr: 1, verticalAlign: 'middle' }} />
                  総合適合度
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <Typography variant="h4" color="primary" sx={{ mr: 2 }}>
                    {(result.overall_compatibility * 100).toFixed(1)}%
                  </Typography>
                  <Chip
                    label={getScoreColor(result.overall_compatibility * 100)}
                    color={getScoreColor(result.overall_compatibility * 100)}
                    size="small"
                  />
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={result.overall_compatibility * 100}
                  color={getScoreColor(result.overall_compatibility * 100)}
                  sx={{ height: 8, borderRadius: 4 }}
                />
              </Box>

              {/* 各特徴量のスコア */}
              <Box sx={{ mb: 3 }}>
                <Typography variant="h6" gutterBottom>特徴量別スコア</Typography>
                <Grid container spacing={2}>
                  {Object.entries(result.feature_scores).map(([feature, score]) => (
                    <Grid item xs={12} sm={6} md={4} key={feature}>
                      <Box>
                        <Typography variant="body2" sx={{ mb: 1 }}>
                          {feature.replace('_', ' ')}
                        </Typography>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <LinearProgress
                            variant="determinate"
                            value={score * 100}
                            color={getScoreColor(score * 100)}
                            sx={{ flexGrow: 1, height: 6, borderRadius: 3 }}
                          />
                          <Typography variant="body2" fontWeight="bold">
                            {(score * 100).toFixed(0)}%
                          </Typography>
                        </Box>
                      </Box>
                    </Grid>
                  ))}
                </Grid>
              </Box>

              {/* 推薦理由 */}
              <Box sx={{ mb: 2, p: 2, backgroundColor: 'grey.50', borderRadius: 2 }}>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  推薦理由
                </Typography>
                <Typography variant="body1">
                  {result.explanation || '詳細な分析結果に基づく推薦です。'}
                </Typography>
              </Box>

              {/* 信頼度 */}
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="body2" color="text.secondary">
                  推薦信頼度: {(result.confidence * 100).toFixed(1)}%
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {result.recommendation}
                </Typography>
              </Box>
            </CardContent>
          </Card>
        ))}
      </Box>
    </Box>
  );
};

export default ResultsList;