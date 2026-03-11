import os
import json
import glob
from datetime import datetime, timezone
import argparse

from strategy_base import ModularStrategy
from signals import RollingMeanSignal, MeanReversionSignal, RegimeSwitchedSignal, MarkovStateSignal
from sizing import MartingaleSizing, ArithmeticSizing, FixedSizing
from execution_simulator import ExecutionSimulator
from tracker import Tracker

def load_data(data_dir: str) -> list:
    """Loads and sorts all 5m candles chronologically."""
    json_files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    all_events = []
    
    for f in json_files:
        with open(f, 'r') as file:
            try:
                events = json.load(file)
                all_events.extend(events)
            except json.JSONDecodeError:
                pass
                
    all_events.sort(key=lambda x: int(x['startTime']))
    return all_events

def run_backtest(events: list, strategy: ModularStrategy, exec_params: dict) -> dict:
    if not events:
        return {}

    simulator = ExecutionSimulator(exec_params)
    tracker = Tracker(initial_capital=exec_params.get('initial_capital', 1000.0))
    
    base_trade_budget = exec_params.get('base_trade_budget', 20.0)
    
    # Session tracking variables
    current_day = None
    daily_trades = 0
    daily_pnl = 0.0
    daily_cons_losses = 0
    session_frozen = False
    
    # Store processed candles for signal generation
    history = []
    
    for i, candle in enumerate(events):
        candle_date = datetime.fromtimestamp(int(candle['startTime'])/1000, tz=timezone.utc).date()
        
        # New session check
        if current_day != candle_date:
            current_day = candle_date
            daily_trades = 0
            daily_pnl = 0.0
            daily_cons_losses = 0
            session_frozen = False
            
        if session_frozen:
            # We must still update history for the signal generator
            history.append(candle)
            if len(history) > 1000:
                history.pop(0)
            continue
            
        # We need a signal based on knowledge UP TO the prior candle
        # The current candle's outcome is not known at startTime
        signal = strategy.get_signal(candle, history)
        
        if signal:
            # We want to enter a trade at the start of this candle
            budget = strategy.get_bet_size(base_trade_budget)
            
            # Constraints: max bet is our remaining capital
            cost = min(budget, tracker.capital)
            
            if cost >= 1.0:
                shares_bought = cost / simulator.entry_price
                
                # Check execution
                if simulator.attempt_entry(signal, candle):
                    # We are in. Now we wait for outcome
                    final_outcome = candle['outcome']
                    
                    # Do we attempt a forced exit?
                    hit_99c = simulator.attempt_forced_exit(signal, candle, final_outcome)
                    
                    pnl = simulator.settle_trade(signal, final_outcome, shares_bought, hit_99c)
                    
                    won = (pnl > 0)
                    
                    dt = datetime.fromtimestamp(int(candle['startTime'])/1000, tz=timezone.utc).isoformat()
                    
                    tracker.record_trade(
                        timestamp=dt,
                        direction=signal,
                        size=shares_bought,
                        pnl=pnl,
                        forced_exit=hit_99c,
                        in_series_step=strategy.series_losses
                    )
                    # Session tracking math
                    daily_trades += 1
                    daily_pnl += pnl
                    if not won:
                        daily_cons_losses += 1
                    else:
                        daily_cons_losses = 0
                        
                    # Check session freezing (Hard Filters)
                    if 'session_params' in exec_params:
                        s_p = exec_params['session_params']
                        max_tr = s_p.get('max_trades_per_day', 999)
                        max_cons_l = s_p.get('max_cons_losses_day', 999)
                        max_risk_units = s_p.get('max_loss_units_day', 999)
                        
                        current_loss_units = -daily_pnl / base_trade_budget if daily_pnl < 0 else 0
                        
                        if daily_trades >= max_tr or daily_cons_losses >= max_cons_l or current_loss_units >= max_risk_units:
                            session_frozen = True
                    
        # Finalize step
        history.append(candle)
        # Cap history to prevent memory bloat, we need 1000 candles max for our longest Markov rules
        if len(history) > 1000:
             history.pop(0)

    # Output report
    report = tracker.generate_report()
    report["execution_params"] = exec_params
    
    return report

if __name__ == "__main__":
    DATA_DIR = "btc_5m_data"
    OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "backtest_results.json")
    
    events = load_data(DATA_DIR)
    
    sig = MarkovStateSignal({})
    size = FixedSizing({})
    strat = ModularStrategy(sig, size, {"max_series_losses": 999})
    
    exec_p = {
        "initial_capital": 100.0,
        "base_trade_budget": 1.0,
        "session_params": {
            "max_trades_per_day": 4,
            "max_cons_losses_day": 2,
            "max_loss_units_day": 2.0
        }
    }
    
    report = run_backtest(events, strat, exec_p)
    
    # Add metadata for the standalone view dashboard
    report["signal"] = "MarkovStateSignal"
    report["signal_params"] = {"state_len": 4, "bias": 0.64, "lookback_500": True, "lookback_200": True}
    report["sizing"] = "Fixed_1Unit"
    report["sizing_params"] = {"base_bet": 1.0}
    report["max_series_losses"] = "Session Limit (2 losses, 4 trades, or -2 units)"
    report["explanation"] = "This strategy uses Markov probabilities: It reads the current 4-result state and looks back 500 outcomes. It only trades if the state appeared 30+ times with a next-outcome probability >= 64%. It also ensures the same side has >= 60% probability in the last 200 outcomes. It skips everything else. It stops trading for the day after 4 trades, -2 units, or 2 consecutive losses."
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(report, f, indent=2)
        
    print(f"Backtest full run complete. Report saved to {OUTPUT_FILE}")
    print(f"Net Profit: ${report['net_profit']} | Win Rate: {report['win_rate_pct']}% | Max DD: {report['max_drawdown_pct']}%")
