from typing import Dict, Any
from strategy_base import SizingBase

class FixedSizing(SizingBase):
    """
    Always bets the base size regardless of streaks.
    """
    def get_bet_size(self, base_size: float, series_losses: int, in_series: bool) -> float:
        return base_size

class MartingaleSizing(SizingBase):
    """
    Multiplies the base size by escalating coefficients after a loss.
    """
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.multipliers = self.params.get('multipliers', [1.0, 2.0, 4.0, 8.0])

    def get_bet_size(self, base_size: float, series_losses: int, in_series: bool) -> float:
        idx = min(series_losses, len(self.multipliers) - 1)
        return base_size * self.multipliers[idx]

class ArithmeticSizing(SizingBase):
    """
    Adds to the base size linearly after a loss (e.g., 1x, 2x, 3x, 4x)
    """
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.multiplier_step = self.params.get('step', 1.0)

    def get_bet_size(self, base_size: float, series_losses: int, in_series: bool) -> float:
        mult = 1.0 + (series_losses * self.multiplier_step)
        return base_size * mult
         
class AntiMartingaleSizing(SizingBase):
    """
    Reduces bet size during losing streaks to protect capital.
    e.g., [1.0, 0.5, 0.25]
    """
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.multipliers = self.params.get('multipliers', [1.0, 0.5, 0.25, 0.125])

    def get_bet_size(self, base_size: float, series_losses: int, in_series: bool) -> float:
        idx = min(series_losses, len(self.multipliers) - 1)
        return base_size * self.multipliers[idx]
