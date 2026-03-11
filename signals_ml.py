import numpy as np
import scipy.stats
from typing import Dict, Any, Optional, List
from signals_advanced import SignalBase
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from hmmlearn import hmm

def extract_sequence_features(outcomes: str, windows: list) -> dict:
    features = {}
    if not outcomes: return features
    
    ups = outcomes.count('U')
    features['recent_up_ratio_all'] = ups / len(outcomes)
    
    current_char = outcomes[-1]
    streak_len = 0
    for i in range(len(outcomes)-1, -1, -1):
        if outcomes[i] == current_char: streak_len += 1
        else: break
        
    features['current_streak_len'] = streak_len
    features['current_streak_dir'] = 1 if current_char == 'U' else 0
    
    longest_streak = 0
    current = 1
    for i in range(1, len(outcomes)):
        if outcomes[i] == outcomes[i-1]:
            current += 1
            longest_streak = max(longest_streak, current)
        else:
            current = 1
            
    features['longest_streak'] = max(longest_streak, current) if len(outcomes) > 0 else 0
    
    flips = sum(1 for i in range(1, len(outcomes)) if outcomes[i] != outcomes[i-1])
    features['flips_ratio'] = flips / max(1, len(outcomes) - 1)
    
    pu = max(0.001, min(0.999, ups / max(1, len(outcomes))))
    features['entropy'] = -(pu * np.log2(pu) + (1-pu) * np.log2(1-pu))
    
    for w in windows:
        if len(outcomes) >= w:
            w_out = outcomes[-w:]
            features[f'up_ratio_{w}'] = w_out.count('U') / w
            w_flips = sum(1 for i in range(1, w) if w_out[i] != w_out[i-1])
            features[f'flips_{w}'] = w_flips
            w_pu = max(0.001, min(0.999, w_out.count('U') / w))
            features[f'entropy_{w}'] = -(w_pu * np.log2(w_pu) + (1-w_pu) * np.log2(1-w_pu))
            
    return features

# ==========================================
# Family 5: Hidden regime model
# ==========================================
class RegimeFeatureSignal(SignalBase):
    def __init__(self, params: dict):
        super().__init__(params)
        self.feature_window = params.get('feature_window', 30)
        self.regime_method = params.get('regime_method', 'kmeans')
        self.n_regimes = params.get('n_regimes', 3)
        self.trend_threshold = params.get('trend_threshold', 0.65)
        self.chop_threshold = params.get('chop_threshold', 0.50)
        self.action_map = params.get('action_map', 'trend_follow_chop_fade')
        self.model = None
        self.fitted = False

    def get_signal(self, current_candle: dict, history: list) -> Optional[str]:
        if len(history) < self.feature_window * 2: return None
        outcomes = "".join(['U' if c['outcome'] == 'up' else 'D' for c in history])
        
        # Fit model periodically or just use rule based if specified
        if self.regime_method == 'rule_based':
            feats = extract_sequence_features(outcomes[-self.feature_window:], [])
            if feats['flips_ratio'] > self.chop_threshold:
                regime = 'chop'
            elif feats['current_streak_len'] >= 3 or abs(feats['recent_up_ratio_all'] - 0.5) > (self.trend_threshold - 0.5):
                regime = 'trend'
            else:
                regime = 'neutral'
                
        else:
            # We would fit K-means/HMM on rolling blocks of feature_window
            # to classify current regime. simplified implementation:
            regime = 'neutral' # stubbed for speed, would build rolling dataset
            
        current_dir = 'up' if outcomes[-1] == 'U' else 'down'
        opp_dir = 'down' if current_dir == 'up' else 'up'
        
        if regime == 'chop':
            if 'chop_fade' in self.action_map: return opp_dir
            if 'chop_skip' in self.action_map: return None
        elif regime == 'trend':
            if 'trend_follow' in self.action_map: return current_dir
            
        return None

# ==========================================
# Family 6: Logistic model
# ==========================================
class LogisticSequenceSignal(SignalBase):
    def __init__(self, params: dict):
        super().__init__(params)
        self.feature_windows = params.get('feature_window_set', [10, 20, 50])
        self.C = params.get('C', 1.0)
        self.pred_threshold = params.get('prediction_threshold', 0.58)
        self.min_prob_gap = params.get('min_prob_gap', 0.10)
        self.model = LogisticRegression(C=self.C, max_iter=200, class_weight=params.get('class_weight', 'balanced'))
        self.min_train_size = 300

    def get_signal(self, current_candle: dict, history: list) -> Optional[str]:
        outcomes = "".join(['U' if c['outcome'] == 'up' else 'D' for c in history])
        if len(outcomes) < self.min_train_size + max(self.feature_windows): return None
        
        # Generate dataset
        # In a real environment, we'd cache this so we aren't re-training per tick
        X, y = [], []
        max_w = max(self.feature_windows)
        for i in range(max_w, len(outcomes)-1, 5): # Step by 5 to speed up
            feats = extract_sequence_features(outcomes[i-max_w:i], self.feature_windows)
            vec = [feats.get(k, 0) for k in sorted(feats.keys())]
            X.append(vec)
            y.append(1 if outcomes[i] == 'U' else 0)
            
        current_feats = extract_sequence_features(outcomes[-max_w:], self.feature_windows)
        X_curr = [[current_feats.get(k, 0) for k in sorted(current_feats.keys())]]
        
        if len(set(y)) < 2: return None
        
        try:
            self.model.fit(X, y)
            probs = self.model.predict_proba(X_curr)[0]
        except:
            return None
            
        p_down, p_up = probs[0], probs[1]
        
        if p_up >= self.pred_threshold and (p_up - p_down) >= self.min_prob_gap: return 'up'
        if p_down >= self.pred_threshold and (p_down - p_up) >= self.min_prob_gap: return 'down'
        return None

# ==========================================
# Family 7: Naive Bayes
# ==========================================
class NaiveBayesSequenceSignal(SignalBase):
    def __init__(self, params: dict):
        super().__init__(params)
        self.feature_windows = params.get('feature_window_set', [10, 20])
        self.alpha = params.get('smoothing_alpha', 1.0)
        self.pred_threshold = params.get('prediction_threshold', 0.58)
        self.model = MultinomialNB(alpha=self.alpha)
        self.max_w = max(self.feature_windows)

    def discretize(self, feats: dict) -> list:
        # NB needs discrete or counts for Multinomial
        d = []
        d.append(min(5, feats.get('current_streak_len', 1)))
        d.append(int(feats.get('flips_ratio', 0) * 10))
        d.append(int(feats.get('entropy', 0) * 10))
        d.append(feats.get('current_streak_dir', 0))
        return d

    def get_signal(self, current_candle: dict, history: list) -> Optional[str]:
        outcomes = "".join(['U' if c['outcome'] == 'up' else 'D' for c in history])
        if len(outcomes) < 200: return None
        
        X, y = [], []
        for i in range(self.max_w, len(outcomes)-1, 3):
            feats = extract_sequence_features(outcomes[i-self.max_w:i], self.feature_windows)
            X.append(self.discretize(feats))
            y.append(1 if outcomes[i] == 'U' else 0)
            
        current_feats = extract_sequence_features(outcomes[-self.max_w:], self.feature_windows)
        X_curr = [self.discretize(current_feats)]
        
        if len(set(y)) < 2: return None
        self.model.fit(X, y)
        probs = self.model.predict_proba(X_curr)[0]
        
        if probs[1] >= self.pred_threshold: return 'up'
        if probs[0] >= self.pred_threshold: return 'down'
        return None

# ==========================================
# Family 8: Random Forest
# ==========================================
class TreeSequenceSignal(SignalBase):
    def __init__(self, params: dict):
        super().__init__(params)
        self.feature_windows = params.get('feature_window_set', [10, 20])
        self.pred_threshold = params.get('prediction_threshold', 0.58)
        self.model = RandomForestClassifier(
            n_estimators=params.get('n_estimators', 50),
            max_depth=params.get('max_depth', 4),
            min_samples_split=params.get('min_samples_split', 30),
            random_state=42
        )
        self.max_w = max(self.feature_windows)

    def get_signal(self, current_candle: dict, history: list) -> Optional[str]:
        outcomes = "".join(['U' if c['outcome'] == 'up' else 'D' for c in history])
        if len(outcomes) < 300: return None
        
        X, y = [], []
        for i in range(self.max_w, len(outcomes)-1, 10): 
            feats = extract_sequence_features(outcomes[i-self.max_w:i], self.feature_windows)
            vec = [feats.get(k, 0) for k in sorted(feats.keys())]
            X.append(vec)
            y.append(1 if outcomes[i] == 'U' else 0)
            
        current_feats = extract_sequence_features(outcomes[-self.max_w:], self.feature_windows)
        X_curr = [[current_feats.get(k, 0) for k in sorted(current_feats.keys())]]
        
        if len(set(y)) < 2: return None
        self.model.fit(X, y)
        probs = self.model.predict_proba(X_curr)[0]
        
        if probs[1] >= self.pred_threshold: return 'up'
        if probs[0] >= self.pred_threshold: return 'down'
        return None
