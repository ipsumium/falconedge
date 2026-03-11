import os
import json
from backtest_engine import load_data, run_backtest
from strategy_base import ModularStrategy
from sizing import FixedSizing

import signals_advanced as sa
import signals_ml as sm
import signals_meta as sme

DATA_DIR = "btc_5m_data"
DISCOVERY_OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "discovery_results.json")
DETAILS_DIR = os.path.join(os.path.dirname(__file__), "discovery_details")

CLASS_MAP = {
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

def evaluate_fold(variant_id, family_name, sig_class, sig_params, risk_params, events_fold, exec_p_base):
    exec_p = dict(exec_p_base)
    exec_p["session_params"] = {
        "max_trades_per_day": risk_params.get("max_trades_day", 4),
        "max_cons_losses_day": risk_params.get("stop_after_losses", 2),
        "max_loss_units_day": float(-risk_params.get("daily_stop", -2))
    }
    
    strat_params = {"max_series_losses": 999}
    strat_params.update(risk_params)
    
    sig = sig_class(sig_params)
    size = FixedSizing({"base_bet": 1.0})
    strat = ModularStrategy(sig, size, strat_params)
    
    return run_backtest(events_fold, strat, exec_p)

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

def generate_explanation(family, sig_params, risk_params):
    desc = FAMILY_DESCRIPTIONS.get(family, f"This strategy belongs to the {family} family.")
    desc += f" It operates with signal parameters such as {', '.join([f'{k}={v}' for k,v in sig_params.items()])}. "
    desc += f"To control risk, it halts trading after {risk_params.get('stop_after_losses', 2)} consecutive daily losses, "
    desc += f"restricts to {risk_params.get('max_trades_day', 4)} max trades per day, and sets a daily stop loss at {-risk_params.get('daily_stop', -2)} units. "
    if risk_params.get('skip_if_streak_too_long_enabled'):
        desc += "It dynamically stops trading after excessive continuous market streaks to avoid severe drawdowns. "
    if risk_params.get('skip_if_recent_dominance_enabled'):
        desc += "It avoids trading during highly dominant one-sided market periods (anti-chop mechanism)."
    return desc

def rebuild():
    print("Loading events...")
    events = load_data(DATA_DIR)
    mid_idx = len(events) // 2
    folds = [
        (events[:mid_idx // 2], events[:mid_idx]), 
        (events[mid_idx // 2 : mid_idx], events[mid_idx:])
    ]
    
    with open(DISCOVERY_OUTPUT_FILE, 'r') as f:
        data = json.load(f)
        
    strategies = data.get("strategies", [])
    
    exec_p_base = {
        "initial_capital": 500.0,
        "base_trade_budget": 5.0
    }
    
    for s in strategies:
        if s["id"].startswith("config_wf_"):
            print(f"Rebuilding {s['id']}...")
            fam = s["signal"]
            sig_class = CLASS_MAP[fam]
            sig_params = s["signal_params"]
            risk_params = s["session_params"]
            
            total_trades = 0
            total_pnl = 0.0
            combined_equity = []
            all_trades = []
            
            fold_pnls = []
            wins = 0
            losses = 0
            
            current_equity = exec_p_base["initial_capital"]
            
            for train_ev, test_ev in folds:
                report = evaluate_fold(s["id"], fam, sig_class, sig_params, risk_params, test_ev, exec_p_base)
                if not report or report['total_trades'] == 0:
                    fold_pnls.append(0)
                    continue
                    
                total_trades += report['total_trades']
                total_pnl += report['net_profit']
                fold_pnls.append(report['net_profit'])
                
                wins += report.get('wins', 0)
                losses += report.get('losses', 0)
                
                # Offset for appending trades and equity
                offset = current_equity - exec_p_base['initial_capital']
                
                if not combined_equity:
                    combined_equity.extend(report['equity_curve'])
                    current_equity = report['equity_curve'][-1]['equity']
                else:
                    for pt in report['equity_curve'][1:]: # skip day 0
                        combined_equity.append({
                            "time": pt["time"],
                            "equity": pt["equity"] + offset,
                            "drawdown": pt["drawdown"]
                        })
                    current_equity = combined_equity[-1]['equity']
                    
                for t in report['trades']:
                    # Adjust cumulative fields? Actually the dashboard calculates from PnL
                    all_trades.append(t)
                    
            # Recalculate Max DD
            max_dd = 0.0
            if combined_equity:
                peak = combined_equity[0]['equity']
                for pt in combined_equity:
                    if pt['equity'] > peak: peak = pt['equity']
                    dd = (peak - pt['equity']) / peak * 100
                    if dd > max_dd: max_dd = dd
            
            # Simple Subsampling of Equity Curve for leaderboard (max 100 points)
            subsampled_equity = []
            if combined_equity:
                step = max(1, len(combined_equity) // 100)
                subsampled_equity = combined_equity[::step]
                
            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
            
            # Update strategy object in leaderboard
            s["net_profit"] = total_pnl
            s["total_trades"] = total_trades
            s["max_dd"] = max_dd
            s["win_rate"] = win_rate
            s["wins"] = wins
            s["losses"] = losses
            s["equity_curve"] = subsampled_equity # Small version for leaderboard
            
            explanation = generate_explanation(fam, sig_params, risk_params)
            
            # Save detail file
            detail_path = os.path.join(DETAILS_DIR, f"{s['id']}.json")
            dash_data = {
                "initial_capital": exec_p_base["initial_capital"],
                "final_capital": exec_p_base["initial_capital"] + total_pnl,
                "net_profit": total_pnl,
                "total_trades": total_trades,
                "win_rate_pct": win_rate,
                "wins": wins,
                "losses": losses,
                "max_drawdown_pct": max_dd,
                "max_consecutive_losses": 0,
                "val_pnl": fold_pnls[0] if len(fold_pnls) > 0 else 0,
                "test_pnl": fold_pnls[1] if len(fold_pnls) > 1 else 0,
                "signal_params": sig_params,
                "session_params": risk_params,
                "explanation": explanation,
                "equity_curve": combined_equity, # Full version for detail page
                "trades": all_trades
            }
            with open(detail_path, 'w') as df:
                json.dump(dash_data, df)
                
    with open(DISCOVERY_OUTPUT_FILE, 'w') as f:
        json.dump(data, f, indent=2)
        
    print("Rebuild complete!")

if __name__ == "__main__":
    rebuild()
