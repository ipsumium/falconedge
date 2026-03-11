from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class SignalBase(ABC):
    """
    Base class for market direction signals.
    """
    def __init__(self, params: Dict[str, Any]):
        self.params = params

    @abstractmethod
    def get_signal(self, current_candle: Dict[str, Any], history: list) -> Optional[str]:
        """
        Returns 'up', 'down', or None (pass).
        """
        pass

class SizingBase(ABC):
    """
    Base class for position sizing logic.
    """
    def __init__(self, params: Dict[str, Any]):
        self.params = params

    @abstractmethod
    def get_bet_size(self, base_size: float, series_losses: int, in_series: bool) -> float:
        """
        Computes the size based on base state and series progression.
        """
        pass

class ModularStrategy:
    """
    Constructs a strategy from discrete Signal and Sizing components.
    """
    
    def __init__(self, signal_component: SignalBase, sizing_component: SizingBase, params: Dict[str, Any]):
        self.signal = signal_component
        self.sizing = sizing_component
        self.params = params
        
        self.series_losses = 0
        self.in_series = False
        self.current_direction = None

    def get_signal(self, current_candle: Dict[str, Any], history: list) -> Optional[str]:
        # Global risk-control overlay
        if len(history) > 0:
            outcomes = "".join(['U' if c['outcome'] == 'up' else 'D' for c in history])
            
            if self.params.get('skip_if_recent_dominance_enabled', False):
                win = self.params.get('recent_dominance_window', 50)
                rat = self.params.get('recent_dominance_ratio', 0.75)
                if len(outcomes) >= win:
                    recent = outcomes[-win:]
                    if recent.count('U') / win >= rat or recent.count('D') / win >= rat:
                        return None
                        
            if self.params.get('skip_if_streak_too_long_enabled', False):
                max_streak = self.params.get('max_live_streak_allowed', 4)
                current_char = outcomes[-1]
                streak = 0
                for i in range(len(outcomes)-1, -1, -1):
                    if outcomes[i] == current_char: streak += 1
                    else: break
                if streak > max_streak:
                    return None

        # Always evaluate the underlying signal module strictly.
        # We no longer force consecutive trades on subsequent candles blindly.
        # If we are in a Martingale series, the escalated size will be applied to the NEXT valid signal.
        return self.signal.get_signal(current_candle, history)

    def get_bet_size(self, base_size: float) -> float:
        return self.sizing.get_bet_size(base_size, self.series_losses, self.in_series)

    def record_outcome(self, won: bool):
        """
        Advances the series state based on the outcome of a settled trade.
        """
        if won:
            # Reset series on win
            self.series_losses = 0
            self.in_series = False
            self.current_direction = None
        else:
            self.series_losses += 1
            self.in_series = True
            
        # Hard stop-loss reset at parameter limit
        if self.series_losses >= self.params.get('max_series_losses', 5):
            self.series_losses = 0
            self.in_series = False
            self.current_direction = None
