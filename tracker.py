import json
from typing import List, Dict, Any

class Tracker:
    """
    Tracks trades, equity curve, max drawdown, and computes metrics.
    """
    
    def __init__(self, initial_capital: float = 1000.0):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.peak_capital = initial_capital
        self.max_drawdown_pct = 0.0
        
        self.trades = []
        self.equity_curve = []
        
        self.wins = 0
        self.losses = 0
        self.consecutive_losses = 0
        self.max_consecutive_losses = 0

    def record_trade(self, timestamp: str, direction: str, size: float, pnl: float, forced_exit: bool, in_series_step: int):
        self.capital += pnl
        
        if self.capital > self.peak_capital:
            self.peak_capital = self.capital
            
        dd_pct = (self.peak_capital - self.capital) / self.peak_capital * 100 if self.peak_capital > 0 else 0
        self.max_drawdown_pct = max(self.max_drawdown_pct, dd_pct)
        
        self.equity_curve.append({
            "time": timestamp,
            "equity": self.capital,
            "drawdown": dd_pct
        })
        
        if pnl > 0:
            self.wins += 1
            self.consecutive_losses = 0
        else:
            self.losses += 1
            self.consecutive_losses += 1
            self.max_consecutive_losses = max(self.max_consecutive_losses, self.consecutive_losses)
            
        self.trades.append({
            "time": timestamp,
            "direction": direction,
            "size_shares": round(size, 2),
            "pnl": round(pnl, 2),
            "forced_exit": forced_exit,
            "series_step": in_series_step
        })

    def generate_report(self) -> Dict[str, Any]:
        total_trades = self.wins + self.losses
        win_rate = (self.wins / total_trades * 100) if total_trades > 0 else 0
        
        return {
            "initial_capital": self.initial_capital,
            "final_capital": round(self.capital, 2),
            "net_profit": round(self.capital - self.initial_capital, 2),
            "total_trades": total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate_pct": round(win_rate, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "max_consecutive_losses": self.max_consecutive_losses,
            "equity_curve": self.equity_curve,
            "trades": self.trades
        }
