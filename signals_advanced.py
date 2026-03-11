import numpy as np
import collections
import math
from typing import Dict, Any, Optional, List

# Machine Learning Imports (used in later families)
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.ensemble import RandomForestClassifier
    from hmmlearn import hmm
    from sklearn.cluster import KMeans
except ImportError:
    pass

class SignalBase:
    """Base class for all advanced signals."""
    def __init__(self, params: dict):
        self.params = params

    def get_signal(self, current_candle: dict, history: list) -> Optional[str]:
        raise NotImplementedError

# ==========================================
# Family 1: Higher-order Markov with Bayesian shrinkage
# ==========================================
class BayesianMarkovSignal(SignalBase):
    def __init__(self, params: dict):
        super().__init__(params)
        self.state_len = params.get('state_len', 4)
        self.lookback = params.get('lookback', 500)
        self.min_samples = params.get('min_samples', 20)
        self.prior_a = params.get('prior_a', 2)
        self.prior_b = params.get('prior_b', 2)
        self.trade_threshold = params.get('trade_threshold', 0.60)
        self.prob_gap_min = params.get('prob_gap_min', 0.16)

    def get_signal(self, current_candle: dict, history: list) -> Optional[str]:
        if len(history) < self.lookback: return None
        outcomes = ['U' if c['outcome'] == 'up' else 'D' for c in history[-self.lookback:]]
        history_str = "".join(outcomes)
        
        current_state = history_str[-self.state_len:]
        u_state = current_state + 'U'
        d_state = current_state + 'D'
        
        u_count = 0
        idx = 0
        while True:
            idx = history_str.find(u_state, idx)
            if idx == -1: break
            u_count += 1
            idx += 1
            
        d_count = 0
        idx = 0
        while True:
            idx = history_str.find(d_state, idx)
            if idx == -1: break
            d_count += 1
            idx += 1
            
        total_samples = u_count + d_count
        if total_samples < self.min_samples: return None
        
        # Bayesian Shrinkage
        posterior_u = (u_count + self.prior_a) / (total_samples + self.prior_a + self.prior_b)
        posterior_d = (d_count + self.prior_b) / (total_samples + self.prior_a + self.prior_b)
        
        if posterior_u >= self.trade_threshold and (posterior_u - posterior_d) >= self.prob_gap_min:
            return 'up'
        if posterior_d >= self.trade_threshold and (posterior_d - posterior_u) >= self.prob_gap_min:
            return 'down'
        return None

# ==========================================
# Family 2: Ensemble of multiple state lengths
# ==========================================
class EnsembleMarkovSignal(SignalBase):
    def __init__(self, params: dict):
        super().__init__(params)
        self.state_set = params.get('state_set', [3, 4, 5])
        self.lookback = params.get('lookback', 500)
        self.min_samples_per_model = params.get('min_samples_per_model', 20)
        self.model_threshold = params.get('model_threshold', 0.60)
        self.vote_mode = params.get('vote_mode', 'majority') # majority, unanimous, weighted_prob, weighted_samples
        self.ensemble_threshold = params.get('ensemble_threshold', 0.60)
        self.min_agreeing_models = params.get('min_agreeing_models', 2)

    def get_signal(self, current_candle: dict, history: list) -> Optional[str]:
        if len(history) < self.lookback: return None
        outcomes = "".join(['U' if c['outcome'] == 'up' else 'D' for c in history[-self.lookback:]])
        
        votes = []
        weighted_probs = []
        total_weights = []
        
        for slen in self.state_set:
            state = outcomes[-slen:]
            u_state = state + 'U'
            d_state = state + 'D'
            
            u_count = sum(1 for i in range(len(outcomes) - slen) if outcomes.startswith(u_state, i))
            d_count = sum(1 for i in range(len(outcomes) - slen) if outcomes.startswith(d_state, i))
            
            tot = u_count + d_count
            if tot >= self.min_samples_per_model:
                pu = u_count / tot
                pd = d_count / tot
                
                if pu >= self.model_threshold:
                    votes.append(('U', pu, tot))
                elif pd >= self.model_threshold:
                    votes.append(('D', pd, tot))
                    
        if len(votes) < self.min_agreeing_models: return None
        
        u_votes = [v for v in votes if v[0] == 'U']
        d_votes = [v for v in votes if v[0] == 'D']
        
        if self.vote_mode == 'unanimous':
            if len(u_votes) == len(votes): return 'up'
            if len(d_votes) == len(votes): return 'down'
            return None
            
        elif self.vote_mode == 'majority':
            if len(u_votes) >= self.min_agreeing_models and len(u_votes) > len(d_votes): return 'up'
            if len(d_votes) >= self.min_agreeing_models and len(d_votes) > len(u_votes): return 'down'
            
        elif self.vote_mode == 'weighted_prob':
            u_vp = sum(v[1] for v in u_votes)
            d_vp = sum(v[1] for v in d_votes)
            if u_vp > d_vp and (u_vp / len(self.state_set)) >= self.ensemble_threshold: return 'up'
            if d_vp > u_vp and (d_vp / len(self.state_set)) >= self.ensemble_threshold: return 'down'
            
        elif self.vote_mode == 'weighted_samples':
            u_vs = sum(v[2] for v in u_votes)
            d_vs = sum(v[2] for v in d_votes)
            if u_vs > d_vs and u_vs > sum(v[2] for v in votes)*self.ensemble_threshold: return 'up'
            if d_vs > u_vs and d_vs > sum(v[2] for v in votes)*self.ensemble_threshold: return 'down'

        return None

# ==========================================
# Family 3: Variable-length context tree
# ==========================================
class ContextTreeSignal(SignalBase):
    def __init__(self, params: dict):
        super().__init__(params)
        self.max_suffix_len = params.get('max_suffix_len', 6)
        self.lookback = params.get('lookback', 800)
        self.threshold = params.get('threshold', 0.60)
        self.fallback_mode = params.get('fallback_mode', 'first_valid')
        
        self.min_samples = {
            7: params.get('min_samples_len_7', 20),
            6: params.get('min_samples_len_6', 22),
            5: params.get('min_samples_len_5', 25),
            4: params.get('min_samples_len_4', 30),
            3: params.get('min_samples_len_3', 35),
            2: 40
        }
        self.prob_gap_min = params.get('prob_gap_min', 0.14)

    def get_signal(self, current_candle: dict, history: list) -> Optional[str]:
        if len(history) < self.lookback: return None
        outcomes = "".join(['U' if c['outcome'] == 'up' else 'D' for c in history[-self.lookback:]])
        
        candidates = []
        for slen in range(self.max_suffix_len, 2, -1):
            state = outcomes[-slen:]
            u_count = sum(1 for i in range(len(outcomes) - slen) if outcomes.startswith(state + 'U', i))
            d_count = sum(1 for i in range(len(outcomes) - slen) if outcomes.startswith(state + 'D', i))
            
            tot = u_count + d_count
            min_req = self.min_samples.get(slen, 30)
            
            if tot >= min_req:
                pu = u_count / tot
                pd = d_count / tot
                if pu >= self.threshold and (pu - pd) >= self.prob_gap_min:
                    candidates.append({'slen': slen, 'dir': 'up', 'prob': pu, 'tot': tot})
                elif pd >= self.threshold and (pd - pu) >= self.prob_gap_min:
                    candidates.append({'slen': slen, 'dir': 'down', 'prob': pd, 'tot': tot})
                    
        if not candidates: return None
        
        if self.fallback_mode == 'first_valid':
            return candidates[0]['dir'] # Longest
        elif self.fallback_mode == 'highest_prob':
            best = max(candidates, key=lambda x: x['prob'])
            return best['dir']
        elif self.fallback_mode == 'highest_samples':
            best = max(candidates, key=lambda x: x['tot'])
            return best['dir']
        elif self.fallback_mode == 'score_based':
            best = max(candidates, key=lambda x: x['prob'] * math.log(x['tot']))
            return best['dir']
        return None

# ==========================================
# Family 4: Recency-weighted transition model
# ==========================================
class RecencyMarkovSignal(SignalBase):
    def __init__(self, params: dict):
        super().__init__(params)
        self.state_len = params.get('state_len', 4)
        self.lookback = params.get('lookback', 500)
        self.decay_lambda = params.get('decay_lambda', 0.01)
        self.min_effective_samples = params.get('min_effective_samples', 15)
        self.trade_threshold = params.get('trade_threshold', 0.60)
        self.prob_gap_min = params.get('prob_gap_min', 0.14)

    def get_signal(self, current_candle: dict, history: list) -> Optional[str]:
        if len(history) < self.lookback: return None
        outcomes = "".join(['U' if c['outcome'] == 'up' else 'D' for c in history[-self.lookback:]])
        
        current_state = outcomes[-self.state_len:]
        
        u_weight = 0.0
        d_weight = 0.0
        
        # We need to iterate over history and apply decay based on distance from present
        n_outcomes = len(outcomes)
        
        idx = 0
        while True:
            idx = outcomes.find(current_state, idx)
            if idx == -1 or idx + self.state_len >= n_outcomes: break
            
            next_char = outcomes[idx + self.state_len]
            # distance from end
            distance = n_outcomes - (idx + self.state_len + 1)
            weight = math.exp(-self.decay_lambda * distance)
            
            if next_char == 'U':
                u_weight += weight
            else:
                d_weight += weight
            idx += 1
            
        tot_weight = u_weight + d_weight
        if tot_weight < self.min_effective_samples: return None
        
        pu = u_weight / tot_weight
        pd = d_weight / tot_weight
        
        if pu >= self.trade_threshold and (pu - pd) >= self.prob_gap_min: return 'up'
        if pd >= self.trade_threshold and (pd - pu) >= self.prob_gap_min: return 'down'
        
        return None

