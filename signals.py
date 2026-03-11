from typing import Dict, Any, Optional
from strategy_base import SignalBase

class RollingMeanSignal(SignalBase):
    """
    A trend-following signal based on majority rule over recent N candles.
    """
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.window = self.params.get('window', 5)

    def get_signal(self, current_candle: Dict[str, Any], history: list) -> Optional[str]:
        if len(history) < self.window:
            return None
            
        recent = history[-self.window:]
        up_count = sum(1 for c in recent if c['outcome'] == 'up')
        down_count = self.window - up_count
        
        if up_count > down_count:
            return 'up'
        elif down_count > up_count:
            return 'down'
        return None

class MomentumSignal(SignalBase):
    """
    Bet in the direction of the immediate previous N candles if they form a streak.
    """
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.streak_length = self.params.get('streak_length', 3)

    def get_signal(self, current_candle: Dict[str, Any], history: list) -> Optional[str]:
        if len(history) < self.streak_length:
            return None
            
        recent = history[-self.streak_length:]
        first_outcome = recent[0]['outcome']
        
        # Check if all recent candles match the first one
        if all(c['outcome'] == first_outcome for c in recent):
            # Ensure we only trigger exactly when the streak reaches the target length
            # by checking if the candle immediately before the streak was different.
            if len(history) > self.streak_length:
                if history[-self.streak_length - 1]['outcome'] == first_outcome:
                    return None
            return first_outcome
            
        return None

class MeanReversionSignal(SignalBase):
    """
    Bet AGAINST the immediate previous N candles if they form a streak.
    """
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.streak_length = self.params.get('streak_length', 3)

    def get_signal(self, current_candle: Dict[str, Any], history: list) -> Optional[str]:
        if len(history) < self.streak_length:
            return None
            
        recent = history[-self.streak_length:]
        first_outcome = recent[0]['outcome']
        
        # Check if all recent candles match the first one
        if all(c['outcome'] == first_outcome for c in recent):
            # Ensure we only trigger exactly when the streak reaches the target length
            if len(history) > self.streak_length:
                if history[-self.streak_length - 1]['outcome'] == first_outcome:
                    return None
            return 'down' if first_outcome == 'up' else 'up'
            
        return None

class RegimeSwitchedSignal(SignalBase):
    """
    Combines Momentum and Mean-Reversion based on 6-candle and 20-candle regimes.
    """
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)

    def get_signal(self, current_candle: Dict[str, Any], history: list) -> Optional[str]:
        if len(history) < 20:
            return None
            
        # Get outcome characters
        outcomes = ['U' if c['outcome'] == 'up' else 'D' for c in history[-20:]]
        history_str = "".join(outcomes)
        
        short_w = history_str[-6:]
        long_w = history_str
        
        u_short = short_w.count('U')
        d_short = short_w.count('D')
        
        u_long = long_w.count('U')
        d_long = long_w.count('D')
        
        # Calculate recent streak length
        streak_len = 0
        streak_side = history_str[-1]
        for i in range(len(history_str)-1, -1, -1):
            if history_str[i] == streak_side:
                streak_len += 1
            else:
                break
                
        # Hard filter 1: Streak too long (>= 4)
        if streak_len >= 4:
            return None
            
        # Hard filter 2: One side >= 15 in last 20
        if u_long >= 15 or d_long >= 15:
            return None
            
        # Momentum logic
        momentum_up = (u_short >= 5) and (u_long >= 12) and (streak_len >= 2) and (streak_side == 'U')
        momentum_down = (d_short >= 5) and (d_long >= 12) and (streak_len >= 2) and (streak_side == 'D')
        
        # Mean Reversion logic
        balanced_short = (u_short in [3, 4] and d_short in [2, 3])
        mr_allowed = balanced_short and (streak_len == 3) and (u_long <= 12 and d_long <= 12)
        
        mr_up = mr_allowed and (streak_side == 'D')
        mr_down = mr_allowed and (streak_side == 'U')
        
        # Priority: Momentum over MeanReversion
        if momentum_up: return 'up'
        if momentum_down: return 'down'
        if mr_up: return 'up'
        if mr_down: return 'down'
        
        return None

class MarkovStateSignal(SignalBase):
    """
    1. Read current 4-result state.
    2. In last 500 outcomes, estimate next-outcome probabilities for that state.
    3. Only continue if: state appeared >= 30 times, one side prob >= 64%.
    4. In last 200 outcomes, require same side prob >= 60%.
    5. Skip everything else.
    """
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.state_len = self.params.get('state_len', 4)
        self.lookback_long = self.params.get('lookback_long', 500)
        self.lookback_short = self.params.get('lookback_short', 200)
        self.min_samples = self.params.get('min_samples', 30)
        self.threshold_long = self.params.get('threshold_long', 0.64)
        self.threshold_short = self.params.get('threshold_short', 0.60)
        self.dominance_n = self.params.get('dominance_n', 50)
        self.dominance_ratio = self.params.get('dominance_ratio', 0.75)

    def get_signal(self, current_candle: Dict[str, Any], history: list) -> Optional[str]:
        if len(history) < self.lookback_long:
            return None
            
        outcomes = ['U' if c['outcome'] == 'up' else 'D' for c in history[-self.lookback_long:]]
        history_str = "".join(outcomes)
        
        # 5. Skip everything else / Dominance check
        if self.dominance_n > 0 and len(history_str) >= self.dominance_n:
            last_n = history_str[-self.dominance_n:]
            cutoff = int(self.dominance_n * self.dominance_ratio)
            if last_n.count('U') >= cutoff or last_n.count('D') >= cutoff:
                return None
            
        # 1. Read current state
        current_state = history_str[-self.state_len:]
        
        # Helper to calc transitions using C-backed find for overlapping matches
        def calc_probs(window_str, state):
            u_state = state + 'U'
            d_state = state + 'D'
            
            u_next = 0
            idx = 0
            while True:
                idx = window_str.find(u_state, idx)
                if idx == -1: break
                u_next += 1
                idx += 1
                
            d_next = 0
            idx = 0
            while True:
                idx = window_str.find(d_state, idx)
                if idx == -1: break
                d_next += 1
                idx += 1
                
            return u_next + d_next, u_next, d_next
            
        # 2 & 3. In lookback_long outcomes
        total_long, u_long, d_long = calc_probs(history_str, current_state)
        
        if total_long < self.min_samples:
            return None
            
        prob_u_long = u_long / total_long
        prob_d_long = d_long / total_long
        
        cand_side = None
        if prob_u_long >= self.threshold_long:
            cand_side = 'U'
        elif prob_d_long >= self.threshold_long:
            cand_side = 'D'
        else:
            return None
            
        # 4. In lookback_short outcomes
        history_short = history_str[-self.lookback_short:]
        total_short, u_short, d_short = calc_probs(history_short, current_state)
        
        if total_short == 0:
            return None
            
        prob_u_short = u_short / total_short
        prob_d_short = d_short / total_short
        
        if cand_side == 'U' and prob_u_short >= self.threshold_short:
            return 'up'
        if cand_side == 'D' and prob_d_short >= self.threshold_short:
            return 'down'
            
        return None
