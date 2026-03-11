import os
import json
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from datetime import datetime

from backtest_engine import load_data, run_backtest
from strategy_base import ModularStrategy
from signals import MarkovStateSignal
from sizing import FixedSizing

DATA_DIR = "btc_5m_data"
DISCOVERY_OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "discovery_results.json")
DETAILS_DIR = os.path.join(os.path.dirname(__file__), "discovery_details")

# --- Search Space ---
# Core logic params (Phase 1)
PHASE_1_SPACE = {
    "state_len": [4, 5],
    "lookback_long": [400, 500, 600, 700],
    "lookback_short": [150, 200, 300],
    "min_samples": [25, 30, 35, 40],
    "threshold_long": [0.62, 0.64, 0.66],
    "threshold_short": [0.58, 0.60, 0.62],
    "dominance_n": [30, 50],
    "dominance_ratio": [0.70, 0.75]
}

# Session params (Phase 2 & OOS)
PHASE_2_SPACE = {
    "max_trades_day": [2, 4, 6],
    "stop_after_losses": [1, 2],
    "daily_stop": [-1, -2, -3]
}

def split_events(events, train_ratio=0.7):
    """Splits events chronologically for walk-forward."""
    split_idx = int(len(events) * train_ratio)
    return events[:split_idx], events[split_idx:]

def evaluate_core_combo(config_id, params, events, exec_p):
    """Phase 1 evaluation ignoring session limits."""
    sig = MarkovStateSignal(params)
    size = FixedSizing({"base_bet": 1.0})
    strat = ModularStrategy(sig, size, {"max_series_losses": 999})
    report = run_backtest(events, strat, exec_p)
    return {
        "id": config_id,
        "params": params,
        "report": report
    }

def evaluate_oos_combo(config_id, params, session_params, sig_name, size_name, max_losses_str, events, exec_p_base):
    exec_p = dict(exec_p_base)
    exec_p["session_params"] = {
        "max_trades_per_day": session_params["max_trades_day"],
        "max_cons_losses_day": session_params["stop_after_losses"],
        "max_loss_units_day": float(-session_params["daily_stop"])
    }
    
    sig = MarkovStateSignal(params)
    size = FixedSizing({"base_bet": 1.0})
    strat = ModularStrategy(sig, size, {"max_series_losses": 999})
    report = run_backtest(events, strat, exec_p)
    
    if not report:
        return None
        
    ret_pct = report['net_profit'] / report['initial_capital']
    dd_pct_decimal = report['max_drawdown_pct'] / 100.0
    calmar = ret_pct / dd_pct_decimal if dd_pct_decimal > 0.001 else (ret_pct * 10)
    
    # Needs to meet minimum viable robustness
    if report['total_trades'] < 10 or report['net_profit'] <= 0:
        return None
        
    summary = {
        "id": config_id,
        "signal": sig_name,
        "signal_params": params,
        "sizing": size_name,
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
    
    detailed_report = {
        "id": config_id,
        "summary": summary,
        "explanation": f"Walk-forward validated Markov Grid. {params}\nSession Lim: {session_params}",
        "execution_params": exec_p,
        "trades": report['trades']
    }
    
    return summary, detailed_report

def run_optimization():
    print("Loading events data into memory...")
    events = load_data(DATA_DIR)
    if not events:
        print("Error: No data found.")
        return

    train_events, oos_events = split_events(events, train_ratio=0.7)
    print(f"Total events: {len(events)} | Train: {len(train_events)} | OOS: {len(oos_events)}")
    
    # Phase 1: Core Params
    core_keys, core_values = zip(*PHASE_1_SPACE.items())
    core_combos = [dict(zip(core_keys, v)) for v in itertools.product(*core_values)]
    
    print(f"\n--- Phase 1: Evaluating {len(core_combos)} core signal variants (Train Data) ---")
    
    exec_p_core = {
        "initial_capital": 100.0,
        "base_trade_budget": 1.0,
        "session_params": {
            "max_trades_per_day": 999,
            "max_cons_losses_day": 999,
            "max_loss_units_day": 999.0
        }
    }
    
    phase_1_results = []
    start_t = time.time()
    
    # Due to very large combination sizes, we use ProcessPoolExecutor
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(
                evaluate_core_combo,
                f"core_{i}", p, train_events, exec_p_core
            ): i for i, p in enumerate(core_combos)
        }
        
        for i, future in enumerate(as_completed(futures)):
            try:
                res = future.result()
                rep = res['report']
                if rep and rep.get('net_profit', 0) > 0 and rep.get('total_trades', 0) >= 20: 
                    # basic viability on training set
                    ret_pct = rep['net_profit'] / rep['initial_capital']
                    dd_pct_decimal = rep['max_drawdown_pct'] / 100.0
                    calmar = ret_pct / dd_pct_decimal if dd_pct_decimal > 0.001 else (ret_pct * 10)
                    res['calmar'] = calmar
                    phase_1_results.append(res)
            except Exception as e:
                pass
            if (i+1) % 100 == 0:
                print(f"Phase 1 Progress: {i+1} / {len(core_combos)}")
                
    print(f"Phase 1 complete in {time.time() - start_t:.1f}s. Survivors: {len(phase_1_results)}")
    
    if not phase_1_results:
        print("No viable core strategies found in Phase 1.")
        return
        
    phase_1_results.sort(key=lambda x: x['calmar'], reverse=True)
    # Take top N core configurations
    top_core = phase_1_results[:20]
    print(f"Selected top {len(top_core)} core configs for Phase 2 OOS Validation.")
    
    # Phase 2: OOS with Risk params
    sess_keys, sess_values = zip(*PHASE_2_SPACE.items())
    sess_combos = [dict(zip(sess_keys, v)) for v in itertools.product(*sess_values)]
    
    print(f"\n--- Phase 2: Evaluating {len(top_core) * len(sess_combos)} risk variants (OOS Data) ---")
    
    exec_p_base = {
        "initial_capital": 100.0,
        "base_trade_budget": 1.0
    }
    
    final_summaries = []
    final_detailed = []
    
    start_t = time.time()
    valid_configs = []
    idx = 0
    for core in top_core:
        for sess in sess_combos:
            valid_configs.append((f"config_markov_{idx}", core['params'], sess))
            idx += 1
            
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(
                evaluate_oos_combo,
                conf[0], conf[1], conf[2], "MarkovStateSignal", "Fixed_1Unit", "Session Limit", oos_events, exec_p_base
            ): i for i, conf in enumerate(valid_configs)
        }
        
        for i, future in enumerate(as_completed(futures)):
            try:
                res = future.result()
                if res:
                    summary, detailed = res
                    final_summaries.append(summary)
                    final_detailed.append(detailed)
            except Exception as e:
                pass
            if (i+1) % 50 == 0:
                print(f"Phase 2 Progress: {i+1} / {len(valid_configs)}")
                
    print(f"Phase 2 complete in {time.time() - start_t:.1f}s. Total OOS Validated: {len(final_summaries)}")
    
    final_summaries.sort(key=lambda x: x['calmar'], reverse=True)
    top_20 = final_summaries[:20]
    
    # Write details to disk
    os.makedirs(DETAILS_DIR, exist_ok=True)
    for det in final_detailed:
        if any(s['id'] == det['id'] for s in top_20):
            detail_path = os.path.join(DETAILS_DIR, f"{det['id']}.json")
            with open(detail_path, 'w') as f:
                json.dump(det, f)
                
    print(f"\n--- Appending Top {len(top_20)} to discovery_results.json ---")
    
    with open(DISCOVERY_OUTPUT_FILE, 'r') as f:
        existing_data = json.load(f)
        
    # Append the top 20
    existing_strategies = existing_data.get("strategies", [])
    # Remove any existing markovs if re-running
    existing_strategies = [s for s in existing_strategies if not s['id'].startswith("config_markov_")]
    existing_strategies.extend(top_20)
    
    # Sort the ENTIRE leaderboard again by Calmar
    existing_strategies.sort(key=lambda x: x['calmar'], reverse=True)
    
    existing_data["strategies"] = existing_strategies
    existing_data["total_tested"] = existing_data.get("total_tested", 0) + len(core_combos) + len(valid_configs)
    existing_data["generated_at"] = datetime.utcnow().isoformat() + "Z"
    
    with open(DISCOVERY_OUTPUT_FILE, 'w') as f:
        json.dump(existing_data, f, indent=2)
        
    print(f"Successfully integrated {len(top_20)} Markov walk-forward strategies to the global leaderboard.")
    print("\nTop 3 Markov Combinations (OOS):")
    for r in top_20[:3]:
        print(f"[{r['id']}] | Profit: ${r['net_profit']} | WinRate: {r['win_rate']}% | Calmar: {r['calmar']}")
        print(f"          Params: {r['signal_params']} | Session: {r['session_params']}")

if __name__ == "__main__":
    run_optimization()
