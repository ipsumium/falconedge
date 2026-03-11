import os
import json
from datetime import datetime

from backtest_engine import load_data, run_backtest
from strategy_base import ModularStrategy
from signals import MarkovStateSignal
from sizing import FixedSizing

DATA_DIR = "btc_5m_data"
DISCOVERY_OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "discovery_results.json")
DETAILS_DIR = os.path.join(os.path.dirname(__file__), "discovery_details")

def run():
    print("Loading events data into memory...")
    events = load_data(DATA_DIR)
    if not events:
        print("Error: No data found.")
        return

    # Specific user config
    signal_params = {
        "state_len": 4,
        "lookback_long": 500,
        "lookback_short": 200,
        "min_samples": 30,
        "threshold_long": 0.64,
        "threshold_short": 0.60,
        "dominance_n": 50,
        "dominance_ratio": 0.72
    }
    
    session_params = {
        "max_trades_per_day": 4,
        "max_cons_losses_day": 2,
        "max_loss_units_day": 2.0
    }
    
    # Scale to standard $500 budget and $5 bet
    exec_p = {
        "initial_capital": 500.0,
        "base_trade_budget": 5.0,
        "session_params": session_params
    }
    
    sig = MarkovStateSignal(signal_params)
    size = FixedSizing({"base_bet": 1.0}) # This uses 1.0 of the base_trade_budget (which is 5.0)
    strat = ModularStrategy(sig, size, {"max_series_losses": 999})
    max_losses_str = "Session Limit (2 loss, 4 trd, -2 units)"
    
    print("Running backtest on full dataset...")
    report = run_backtest(events, strat, exec_p)
    
    if not report:
        print("Strategy returned no trades or failed.")
        return
        
    ret_pct = report['net_profit'] / report['initial_capital']
    dd_pct_decimal = report['max_drawdown_pct'] / 100.0
    calmar = ret_pct / dd_pct_decimal if dd_pct_decimal > 0.001 else (ret_pct * 10)
    
    config_id = "config_custom_markov_1"
    
    summary = {
        "id": config_id,
        "signal": "MarkovStateSignal",
        "signal_params": signal_params,
        "sizing": "Fixed_1Unit",
        "sizing_params": {"base_bet": 1.0},
        "session_params": session_params,
        "max_series_losses": max_losses_str,
        
        "net_profit": report['net_profit'],
        "win_rate": report['win_rate_pct'],
        "max_dd": report['max_drawdown_pct'],
        "total_trades": report['total_trades'],
        "calmar": round(calmar, 3),
        "equity_curve": report['equity_curve'],
    }
    
    detailed = {
        "id": config_id,
        "summary": summary,
        "explanation": "This strategy uses Markov probabilities: It reads the current 4-result state and looks back 500 outcomes. It only trades if the state appeared 30+ times with a next-outcome probability >= 64%. It also ensures the same side has >= 60% probability in the last 200 outcomes. It skips everything else. It stops trading for the day after 4 trades, -2 units, or 2 consecutive losses.",
        "execution_params": exec_p,
        "trades": report['trades']
    }
    
    print(f"Profit: ${summary['net_profit']} | WinRate: {summary['win_rate']}% | Trades: {summary['total_trades']} | Max DD: {summary['max_dd']}%")
    
    # Save detail JSON
    os.makedirs(DETAILS_DIR, exist_ok=True)
    detail_path = os.path.join(DETAILS_DIR, f"{config_id}.json")
    with open(detail_path, 'w') as f:
        json.dump(detailed, f)
        
    # Append to leaderboard
    with open(DISCOVERY_OUTPUT_FILE, 'r') as f:
        existing_data = json.load(f)
        
    strategies = existing_data.get("strategies", [])
    # Remove previous version if exists
    strategies = [s for s in strategies if s['id'] != config_id]
    
    strategies.append(summary)
    strategies.sort(key=lambda x: x['calmar'], reverse=True)
    
    existing_data["strategies"] = strategies
    
    with open(DISCOVERY_OUTPUT_FILE, 'w') as f:
        json.dump(existing_data, f, indent=2)
        
    print(f"Saved to {config_id}.json and appended to leaderboard.")

if __name__ == "__main__":
    run()
