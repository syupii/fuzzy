// frontend/src/App.tsx - シンプル版
import React, { useState } from 'react';
import {
  Box,
  Container,
  Typography,
  Paper,
  Tabs,
  Tab,
  Alert,
  AppBar,
  Toolbar
} from '@mui/material';
import { Psychology, Science } from '@mui/icons-material';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';

import EvaluationForm from './components/EvaluationForm';
import ResultsList from './components/ResultsList';
import { EvaluationResponse } from './services/api';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#2196f3',
    },
    secondary: {
      main: '#ff9800',
    },
  },
});

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

const App: React.FC = () => {
  const [currentTab, setCurrentTab] = useState(0);
  const [results, setResults] = useState<EvaluationResponse | null>(null);
  const [error, setError] = useState<string>('');

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setCurrentTab(newValue);
  };

  const handleEvaluationResults = (response: EvaluationResponse) => {
    setResults(response);
    setCurrentTab(1);
    setError('');
  };

  const handleEvaluationError = (errorMessage: string) => {
    setError(errorMessage);
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ flexGrow: 1 }}>
        <AppBar position="static">
          <Toolbar>
            <Science sx={{ mr: 2 }} />
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              研究室選択支援システム
            </Typography>
          </Toolbar>
        </AppBar>

        <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
          {error && (
            <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError('')}>
              {error}
            </Alert>
          )}

          <Paper sx={{ width: '100%' }}>
            <Tabs
              value={currentTab}
              onChange={handleTabChange}
              indicatorColor="primary"
              textColor="primary"
              variant="fullWidth"
            >
              <Tab
                label="評価設定"
                icon={<Psychology />}
                iconPosition="start"
              />
              <Tab
                label={
                  results
                    ? `評価結果 (${(results.results || results.lab_results || []).length}件)`
                    : "評価結果"
                }
                icon={<Science />}
                iconPosition="start"
                disabled={!results}
              />
            </Tabs>

            <TabPanel value={currentTab} index={0}>
              <EvaluationForm
                onResults={handleEvaluationResults}
                onError={handleEvaluationError}
              />
            </TabPanel>

            <TabPanel value={currentTab} index={1}>
              {results ? (
                <ResultsList data={results} />
              ) : (
                <Box sx={{ textAlign: 'center', py: 8 }}>
                  <Science sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
                  <Typography variant="h6" color="text.secondary" gutterBottom>
                    評価結果がありません
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    「評価設定」タブで条件を設定し、評価を実行してください
                  </Typography>
                </Box>
              )}
            </TabPanel>
          </Paper>
        </Container>
      </Box>
    </ThemeProvider>
  );
};

export default App;