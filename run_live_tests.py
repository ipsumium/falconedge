import os
import json
import glob
from datetime import datetime, timezone

from backtest_engine import run_backtest
from strategy_base import ModularStrategy
from sizing import FixedSizing, MartingaleSizing, AntiMartingaleSizing, ArithmeticSizing

import signals as base_s
import signals_advanced as sa
import signals_ml as sm
import signals_meta as sme

DATA_DIR = "btc_5m_data"
BASE_DIR = "/Users/zmeura/Documents/polymarket/FalconEdge"
RESULTS_FILE = os.path.join(BASE_DIR, "discovery_results.json")

CLASS_MAP = {
    "RollingMean": base_s.RollingMeanSignal,
    "Momentum": base_s.MomentumSignal,
    "MeanReversion": base_s.MeanReversionSignal,
    "MarkovStateSignal": base_s.MarkovStateSignal,
    "Family_1": sa.BayesianMarkovSignal,
    "Family_2": sa.EnsembleMarkovSignal,
    "Family_3": sa.ContextTreeSignal,
    "Family_4": sa.RecencyMarkovSignal,
    "Family_5": sm.RegimeFeatureSignal,
    "Family_6": sm.LogisticSequenceSignal,
    "Family_7": sm.NaiveBayesSequenceSignal,
    "Family_8": sm.TreeSequenceSignal,
    "Family_9": sme.MetaLabelSignal,
    "Family_10": sme.BanditSelectorSignal,
    "Family_11": sme.StreakStructureSignal,
    "Family_12": sme.CompressionNoveltySignal,
}

FAMILY_DESCRIPTIONS = {
    "Family_1": "Bayesian Markov - This model uses a Markov Chain enhanced with Dirichlet priors to smooth transition probabilities, preventing overconfidence in rare state transitions.",
    "Family_2": "Ensemble Markov - This model evaluates an ensemble of multiple Markov Chains with varying state lengths, using a voting mechanism to generate robust signals.",
    "Family_3": "Context Tree - This model acts as a variable-order Markov chain, looking back dynamically depending on the detected market context (Context Tree Weighting approach).",
    "Family_4": "Recency Markov - This model applies an exponential decay to historical observations, prioritizing the most recent price action over old data.",
    "Family_5": "Regime Feature - This model calculates rolling volatility and momentum to identify whether the market is trending or chopping, adapting its actions accordingly.",
    "Family_6": "Logistic Sequence - This is a Machine Learning approach using Logistic Regression trained on sequence features (like streak counts and recent directions) to predict the next move.",
    "Family_7": "Naive Bayes Sequence - This is a Machine Learning approach using a Naive Bayes classifier trained on recent sequence features, applying Laplace smoothing to estimate outcome probabilities.",
    "Family_8": "Tree Sequence - This is a Machine Learning approach using Decision Trees & Random Forests trained on sequence features to capture non-linear market patterns.",
    "Family_9": "Meta Labeling - This model acts as a secondary filter. It observes a base strategy and predicts whether the base strategy's next trade will be a win or a loss.",
    "Family_10": "Bandit Selector - This model uses a Multi-Armed Bandit algorithm that treats different actions as 'arms'. It dynamically allocates capital to the best-performing action based on a rolling reward window.",
    "Family_11": "Streak Structure - This model identifies specific sequences of consecutive wins/losses and selectively fades or follows them based on historical distributions.",
    "Family_12": "Compression Novelty - This algorithm acts as a novelty detector that identifies how rare or common the current market sequence is, trading based on whether the market is repeating familiar patterns or entering novel territory."
}

def generate_full_explanation(s):
    sig_name = s.get("signal", "")
    sig_params = s.get("signal_params", {})
    size_name = s.get("sizing", "")
    size_params = s.get("sizing_params", {})
    risk_params = s.get("session_params", {})
    max_losses = s.get("max_series_losses", 999)
    
    if "Family_" in sig_name:
        desc = FAMILY_DESCRIPTIONS.get(sig_name, f"This strategy belongs to the {sig_name} family.")
        desc += f" It operates with signal parameters such as {', '.join([f'{k}={v}' for k,v in sig_params.items()])}. "
        desc += f"To control risk, it halts trading after {risk_params.get('stop_after_losses', 2)} consecutive daily losses, "
        desc += f"restricts to {risk_params.get('max_trades_day', 4)} max trades per day, and sets a daily stop loss at {-risk_params.get('daily_stop', -2)} units. "
        if risk_params.get('skip_if_streak_too_long_enabled'):
            desc += "It dynamically stops trading after excessive continuous market streaks to avoid severe drawdowns. "
        if risk_params.get('skip_if_recent_dominance_enabled'):
            desc += "It avoids trading during highly dominant one-sided market periods (anti-chop mechanism)."
        return desc
    else:
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

SIZING_MAP = {
    "Fixed": FixedSizing,
    "Martingale_Moderate": MartingaleSizing,
    "Martingale_Soft": MartingaleSizing,
    "Martingale_Aggressive": MartingaleSizing,
    "Martingale_Extreme": MartingaleSizing,
    "Arithmetic": ArithmeticSizing,
    "Arithmetic_Soft": ArithmeticSizing,
    "Arithmetic_Aggressive": ArithmeticSizing,
    "AntiMartingale": AntiMartingaleSizing,
    "AntiMartingale_Aggressive": AntiMartingaleSizing,
    "Fixed_1Unit": FixedSizing,
    "Fixed_1Unit_WF": FixedSizing,
}

def load_all_events():
    path_pattern = os.path.join(DATA_DIR, "*.json")
    files = sorted(glob.glob(path_pattern))
    
    all_events = []
    for path in files:
        # Don't load anything past March 9 to avoid future leaks if present
        if os.path.basename(path) > "2026-03-09.json":
            continue
        with open(path, 'r') as f:
            events = json.load(f)
        for e in events:
            if 'maxPrice' in e and 'minPrice' in e:
                 e['highPrice'] = e['maxPrice']
                 e['lowPrice'] = e['minPrice']
            if e.get("openPrice") and e.get("closePrice"):
                all_events.append(e)
    return sorted(all_events, key=lambda x: int(x["startTime"]))[-2000:]

def evaluate_all(strategy_config, events_full, exec_p_base):
    fam = strategy_config["signal"]
    sig_class = CLASS_MAP[fam]
    sig_params = strategy_config.get("signal_params", {})
    risk_params = strategy_config.get("session_params", {})
    
    exec_p = dict(exec_p_base)
    exec_p["session_params"] = {
        "max_trades_per_day": risk_params.get("max_trades_day", 9999),
        "max_cons_losses_day": risk_params.get("stop_after_losses", 9999),
        "max_loss_units_day": float(-risk_params.get("daily_stop", -9999))
    }
    
    strat_params = {"max_series_losses": strategy_config.get("max_series_losses", 999)}
    strat_params.update(risk_params)
    
    sig = sig_class(sig_params)
    
    size_str = strategy_config.get("sizing", "Fixed_1Unit")
    size_class = SIZING_MAP.get(size_str, FixedSizing)
    size_params = strategy_config.get("sizing_params", {})
    # If legacy strategy uses sizing_params = {} but size_class needs config, Sizing base handles {} ok
    
    size = size_class(size_params)
    strat = ModularStrategy(sig, size, strat_params)
    
    return run_backtest(events_full, strat, exec_p)

def process_day_trades(trades, target_date_str):
    day_trades = []
    for t in trades:
        # t['time'] is like '2026-01-01T16:40:00+00:00'
        if t['time'][:10] == target_date_str:
            day_trades.append(t)
            
    # Rebuild equity curve and stats
    cap = 500.0
    peak = 500.0
    max_dd = 0.0
    wins = 0
    losses = 0
    net_profit = 0.0
    
    equity_curve = []
    
    # Starting point before any trades on this day
    if day_trades:
        # Use target day's absolute start string e.g., '2026-03-07T00:00:00+00:00'
        first_time = f"{target_date_str}T00:00:00+00:00"
        equity_curve.append({"time": first_time, "equity": cap, "drawdown": 0})
        
    for t in day_trades:
        pnl = t['pnl']
        cap += pnl
        net_profit += pnl
        if cap > peak: peak = cap
        
        dd = (peak - cap) / peak * 100 if peak > 0 else 0
        if dd > max_dd: max_dd = dd
        
        if pnl >= 0: wins += 1
        else: losses += 1
        
        equity_curve.append({
            "time": t['time'],
            "equity": cap,
            "drawdown": dd
        })
        
    total_trades = wins + losses
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    
    return {
        "trades": day_trades,
        "equity_curve": equity_curve,
        "net_profit": net_profit,
        "wins": wins,
        "losses": losses,
        "total_trades": total_trades,
        "max_dd": max_dd,
        "win_rate": win_rate
    }

def generate_live_tests():
    print("Loading original discovery results...")
    with open(RESULTS_FILE, 'r') as f:
        data = json.load(f)
        
    strategies = data.get("strategies", [])
    profitable = [s for s in strategies if s.get("calmar", 0) > 0 and s.get("signal") in CLASS_MAP]
    top_500 = sorted(profitable, key=lambda x: x.get("calmar", 0), reverse=True)[:500]
    
    print(f"Loaded top {len(top_500)} backtest strategies.")
    
    exec_p_base = {
        "initial_capital": 500.0,
        "base_trade_budget": 5.0
    }
    
    print("Loading all historic events up to March 9 to allow correct signal warmup...")
    events = load_all_events()
    print(f"Total events loaded: {len(events)}")
    
    days_to_extract = ["2026-03-07", "2026-03-08", "2026-03-09"]
    all_day_results = { d: [] for d in days_to_extract }
    
    # We will only evaluate each strategy ONCE
    for i, s in enumerate(top_500):
        if i % 50 == 0: print(f"  Evaluating strat {i}/{len(top_500)}...")
        
        report = evaluate_all(s, events, exec_p_base)
        if not report: continue
        
        all_trades = report.get('trades', [])
        
        # Now segment by day
        for day in days_to_extract:
            day_stats = process_day_trades(all_trades, day)
            
            # Subsample equity curve for leaderboard display
            subsampled_equity = []
            if day_stats['equity_curve']:
                step = max(1, len(day_stats['equity_curve']) // 100)
                subsampled_equity = day_stats['equity_curve'][::step]
            
            live_s = dict(s)
            live_s["id"] = f"live_{day}_{s['id']}"
            live_s["net_profit"] = day_stats['net_profit']
            live_s["total_trades"] = day_stats['total_trades']
            live_s["max_dd"] = day_stats['max_dd']
            live_s["win_rate"] = day_stats['win_rate']
            live_s["wins"] = day_stats['wins']
            live_s["losses"] = day_stats['losses']
            live_s["explanation"] = generate_full_explanation(live_s)
            live_s["equity_curve"] = subsampled_equity
            
            # Simple fallback calmar logic for the single-day subset
            total_pnl = day_stats['net_profit']
            max_dd = day_stats['max_dd']
            live_s["calmar"] = (total_pnl / 100.0) / (max_dd / 100.0) if max_dd > 0.001 else (total_pnl / 10.0)
            
            # We add it even if zero trades so we can track the exact 500 config pool, 
            # or optionally sort it at the end out of view.
            all_day_results[day].append(live_s)
            
            # Create detailed json
            detail_path = os.path.join(BASE_DIR, "discovery_details", f"{live_s['id']}.json")
            dash_data = {
                "initial_capital": exec_p_base["initial_capital"],
                "final_capital": exec_p_base["initial_capital"] + total_pnl,
                "net_profit": total_pnl,
                "total_trades": day_stats['total_trades'],
                "win_rate_pct": day_stats['win_rate'],
                "wins": day_stats['wins'],
                "losses": day_stats['losses'],
                "max_drawdown_pct": max_dd,
                "max_consecutive_losses": 0,
                "signal": s.get("signal", ""),
                "signal_params": s.get("signal_params", {}),
                "sizing": s.get("sizing", ""),
                "sizing_params": s.get("sizing_params", {}),
                "session_params": s.get("session_params", {}),
                "explanation": generate_full_explanation(live_s),
                "equity_curve": day_stats['equity_curve'],
                "trades": day_stats['trades']
            }
            with open(detail_path, 'w') as df:
                json.dump(dash_data, df)
                
    for day in days_to_extract:
        out_file = os.path.join(BASE_DIR, f"live_results_{day}.json")
        sorted_results = sorted(all_day_results[day], key=lambda x: x.get('net_profit', 0), reverse=True)
        out_data = {
            "title": f"Live Data Test: {day}",
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "total_tested": len(top_500),
            "strategies": sorted_results
        }
        with open(out_file, 'w') as f:
            json.dump(out_data, f, indent=2)
            
    print("Done regenerating live tests with full history context!")

if __name__ == "__main__":
    generate_live_tests()
