import os
import json
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from datetime import datetime

from backtest_engine import load_data, run_backtest
from strategy_base import ModularStrategy
from signals import RollingMeanSignal, MomentumSignal, MeanReversionSignal
from sizing import FixedSizing, MartingaleSizing, AntiMartingaleSizing, ArithmeticSizing

DATA_DIR = "btc_5m_data"
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "discovery_results.json")
DETAILS_DIR = os.path.join(os.path.dirname(__file__), "discovery_details")

def generate_explanation(sig_name, sig_params, size_name, size_params, max_losses):
    explanation = f"This strategy uses a '{sig_name}' signal: "
    if "RollingMean" in sig_name:
        explanation += f"It looks at the last {sig_params.get('window')} candles and bets in the direction of the majority."
    elif "Momentum" in sig_name:
        explanation += f"It waits for a streak of {sig_params.get('streak_length')} identical outcomes and bets that the streak will continue."
    elif "MeanReversion" in sig_name:
        explanation += f"It waits for a streak of {sig_params.get('streak_length')} identical outcomes and bets AGAINST the streak, expecting a reversal."
        
    explanation += f"\nFor sizing, it uses a '{size_name}' approach: "
    if "Fixed" in size_name:
        explanation += "It always bets a flat base amount on every trade."
    elif "Martingale" in size_name and "Anti" not in size_name:
        explanation += f"After a loss, it multiplies its bet size by the sequence {size_params.get('multipliers')} to recover previous losses."
    elif "AntiMartingale" in size_name:
        explanation += f"After a loss, it reduces its bet size by the sequence {size_params.get('multipliers')} to protect capital during losing streaks."
    elif "Arithmetic" in size_name:
        explanation += f"After a loss, it linearly increases its bet size by adding {size_params.get('step')}x to the multiplier."
        
    explanation += f"\nIf it suffers {max_losses} losses in a row, the streak resets to avoid complete bankruptcy."
    return explanation

# Hyperparameter Space Definition
SIGNAL_SPACE = [
    ("RollingMean", RollingMeanSignal, {"window": w}) for w in [1, 2, 3, 5, 8, 12, 15, 20, 30, 45, 60]
] + [
    ("Momentum", MomentumSignal, {"streak_length": s}) for s in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]
] + [
    ("MeanReversion", MeanReversionSignal, {"streak_length": s}) for s in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]
]

SIZING_SPACE = [
    ("Fixed", FixedSizing, {}),
    ("Martingale_Moderate", MartingaleSizing, {"multipliers": [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]}),
    ("Martingale_Soft", MartingaleSizing, {"multipliers": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]}),
    ("Martingale_Aggressive", MartingaleSizing, {"multipliers": [1.0, 2.0, 4.0, 8.0, 16.0]}),
    ("Martingale_Extreme", MartingaleSizing, {"multipliers": [1.0, 3.0, 9.0]}),
    ("Arithmetic", ArithmeticSizing, {"step": 1.0}),
    ("Arithmetic_Soft", ArithmeticSizing, {"step": 0.5}),
    ("Arithmetic_Aggressive", ArithmeticSizing, {"step": 2.0}),
    ("AntiMartingale", AntiMartingaleSizing, {"multipliers": [1.0, 0.5, 0.25, 0.125]}),
    ("AntiMartingale_Aggressive", AntiMartingaleSizing, {"multipliers": [1.0, 0.25, 0.05, 0.0]})
]

MAX_LOSSES_SPACE = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12]

EXECUTION_PARAMS = {
    "initial_capital": 500.0,
    "base_trade_budget": 5.0
}

def evaluate_combination(config_id, signal_tuple, sizing_tuple, max_losses, events):
    """
    Evaluates a single parameter combination.
    """
    sig_name, sig_class, sig_params = signal_tuple
    size_name, size_class, size_params = sizing_tuple
    
    # Initialize components
    sig_instance = sig_class(sig_params)
    size_instance = size_class(size_params)
    
    # Initialize strategy wrapper
    strat_params = {"max_series_losses": max_losses}
    strategy = ModularStrategy(sig_instance, size_instance, strat_params)
    
    # Run
    report = run_backtest(events, strategy, EXECUTION_PARAMS)
    
    if not report:
        return None
        
    # Calculate Calmar Ratio proxy (Assume data spans roughly N weeks, here we just use Net Profit / Max DD)
    # Using an absolute Calmar proxy: (Net Profit / Initial Capital) / (Max DD % / 100)
    ret_pct = report['net_profit'] / report['initial_capital']
    dd_pct_decimal = report['max_drawdown_pct'] / 100.0
    
    # Avoid zero division
    calmar = ret_pct / dd_pct_decimal if dd_pct_decimal > 0.001 else (ret_pct * 10) 
    
    summary = {
        "id": config_id,
        "signal": sig_name,
        "signal_params": sig_params,
        "sizing": size_name,
        "sizing_params": size_params,
        "max_series_losses": max_losses,
        
        "net_profit": report['net_profit'],
        "win_rate": report['win_rate_pct'],
        "max_dd": report['max_drawdown_pct'],
        "total_trades": report['total_trades'],
        "calmar": round(calmar, 3),
        
        # Keep full curve data only for the frontend parsing
        "equity_curve": report['equity_curve'],
        # Truncate trades to save massive JSON file sizes across hundreds of combinations
        # "trades" is omitted in discovery grid unless explicitly drilled into later.
    }
    
    # Save the detailed report individually
    detailed_report = {
        "id": config_id,
        "summary": summary,
        "explanation": generate_explanation(sig_name, sig_params, size_name, size_params, max_losses),
        "execution_params": EXECUTION_PARAMS,
        "trades": report['trades']
    }
    
    os.makedirs(DETAILS_DIR, exist_ok=True)
    detail_path = os.path.join(DETAILS_DIR, f"{config_id}.json")
    with open(detail_path, 'w') as f:
        json.dump(detailed_report, f)
        
    return summary

def run_discovery(events):
    combinations = []
    for sig in SIGNAL_SPACE:
        for size in SIZING_SPACE:
            sizing_params = size[2]
            if "multipliers" in sizing_params:
                # Sync max losses EXACTLY with the length of the multiplier sequence
                max_losses = len(sizing_params["multipliers"])
                combinations.append((sig, size, max_losses))
            else:
                # For algorithms without predefined multipliers (Arithmetic, Fixed), sweep bounds manually
                for ml in MAX_LOSSES_SPACE:
                    combinations.append((sig, size, ml))
    print(f"Combinatorial search space size: {len(combinations)} configurations.")
    
    results = []
    start_t = time.time()
    
    # We use ProcessPoolExecutor to parallelize CPU-bound backtest simulations
    # Note: passing `events` to workers uses memory, but keeping it small prevents overhead
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(
                evaluate_combination, 
                f"config_{i}", 
                combo[0], combo[1], combo[2], 
                events
            ): i for i, combo in enumerate(combinations)
        }
        
        for i, future in enumerate(as_completed(futures)):
            try:
                res = future.result()
                if res:
                    results.append(res)
            except Exception as e:
                print(f"Error evaluating config: {e}")
                
            if (i+1) % 10 == 0:
                print(f"Progress: {i+1} / {len(combinations)}")

    elapsed = time.time() - start_t
    print(f"Discovery complete in {elapsed:.1f} seconds.")
    
    # Sort results by Calmar ratio descending (Risk Adjusted Return)
    results.sort(key=lambda x: x['calmar'], reverse=True)
    
    # Save results
    with open(OUTPUT_FILE, 'w') as f:
        json.dump({
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "execution_params": EXECUTION_PARAMS,
            "total_tested": len(combinations),
            "strategies": results
        }, f, indent=2)
        
    print(f"Saved {len(results)} evaluated strategies to {OUTPUT_FILE}")
    
    print("\n--- TOP 3 STRATEGIES (By Calmar Ratio) ---")
    for r in results[:3]:
        print(f"[{r['id']}] Signal: {r['signal']}({r['signal_params']}) | Sizing: {r['sizing']} | TL: {r['max_series_losses']}")
        print(f"      Profit: ${r['net_profit']} | WinRate: {r['win_rate']}% | MaxDD: {r['max_dd']}% | Calmar: {r['calmar']}\n")

if __name__ == "__main__":
    print("Loading events data into memory...")
    events = load_data(DATA_DIR)
    
    if not events:
        print("Error: No data found to run discovery.")
    else:
        run_discovery(events)
