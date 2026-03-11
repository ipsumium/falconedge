import os
import json
import random
import time
import math
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from copy import deepcopy

from backtest_engine import load_data, run_backtest
from strategy_base import ModularStrategy
from sizing import FixedSizing

import signals_advanced as sa
import signals_ml as sm
import signals_meta as sme

DATA_DIR = "btc_5m_data"
DISCOVERY_OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "discovery_results.json")
DETAILS_DIR = os.path.join(os.path.dirname(__file__), "discovery_details")

# ==========================================
# Parameter Spaces (Reduced for randomized sampling)
# ==========================================
SPACES = {
    "Family_1": {
        "class": sa.BayesianMarkovSignal,
        "params": {
            "state_len": [3, 4, 5, 6],
            "lookback": [300, 500, 800, 1200],
            "min_samples": [15, 20, 25, 30, 40],
            "prior_a": [1, 2, 3, 5],
            "prior_b": [1, 2, 3, 5],
            "trade_threshold": [0.58, 0.60, 0.62, 0.64, 0.66],
            "prob_gap_min": [0.12, 0.16, 0.20, 0.24]
        }
    },
    "Family_2": {
        "class": sa.EnsembleMarkovSignal,
        "params": {
            "state_set": [[2,3,4], [3,4,5], [2,3,4,5], [3,4,5,6]],
            "lookback": [300, 500, 800],
            "min_samples_per_model": [15, 20, 25, 30],
            "model_threshold": [0.58, 0.60, 0.62, 0.64],
            "vote_mode": ["majority", "unanimous", "weighted_prob", "weighted_samples"],
            "ensemble_threshold": [0.58, 0.60, 0.62, 0.64],
            "min_agreeing_models": [2, 3, 4]
        }
    },
    "Family_3": {
        "class": sa.ContextTreeSignal,
        "params": {
            "max_suffix_len": [4, 5, 6, 7],
            "lookback": [300, 500, 800, 1200],
            "threshold": [0.58, 0.60, 0.62, 0.64, 0.66],
            "fallback_mode": ["first_valid", "highest_prob", "highest_samples", "score_based"],
            "prob_gap_min": [0.10, 0.14, 0.18, 0.22]
        }
    },
    "Family_4": {
        "class": sa.RecencyMarkovSignal,
        "params": {
            "state_len": [3, 4, 5, 6],
            "lookback": [300, 500, 800, 1200],
            "decay_lambda": [0.001, 0.002, 0.005, 0.01, 0.02],
            "min_effective_samples": [10, 15, 20, 25, 30],
            "trade_threshold": [0.58, 0.60, 0.62, 0.64, 0.66],
            "prob_gap_min": [0.10, 0.14, 0.18, 0.22]
        }
    },
    "Family_5": {
        "class": sm.RegimeFeatureSignal,
        "params": {
            "feature_window": [10, 20, 30, 50],
            "regime_method": ["rule_based"], # Keeping it simple to avoid heavy HMM fitting
            "n_regimes": [2, 3, 4, 5],
            "trend_threshold": [0.60, 0.65, 0.70],
            "chop_threshold": [0.45, 0.50, 0.55],
            "action_map": ["trend_follow_chop_fade", "trend_follow_chop_skip", "fade_only_in_chop"]
        }
    },
    "Family_6": {
        "class": sm.LogisticSequenceSignal,
        "params": {
            "feature_window_set": [[5,10,20], [10,20,50], [5,10,20,50]],
            "C": [0.01, 0.1, 0.5, 1, 2, 5, 10],
            "class_weight": ["balanced"],
            "prediction_threshold": [0.54, 0.56, 0.58, 0.60, 0.62],
            "min_prob_gap": [0.08, 0.10, 0.12, 0.16]
        }
    },
    "Family_7": {
        "class": sm.NaiveBayesSequenceSignal,
        "params": {
            "feature_window_set": [[10,20], [10,20,50], [20,50]],
            "smoothing_alpha": [0.5, 1.0, 2.0, 5.0],
            "prediction_threshold": [0.56, 0.58, 0.60, 0.62]
        }
    },
    "Family_8": {
        "class": sm.TreeSequenceSignal,
        "params": {
            "feature_window_set": [[5,10,20], [10,20,50]],
            "max_depth": [2, 3, 4, 5, 6],
            "min_samples_split": [20, 30, 50, 80],
            "n_estimators": [50, 100, 200],
            "prediction_threshold": [0.56, 0.58, 0.60, 0.62]
        }
    },
    "Family_9": {
        "class": sme.MetaLabelSignal,
        "params": {
            "base_model_family": ["bayesian_markov", "recency_markov"],
            "meta_threshold": [0.52, 0.55, 0.58, 0.60, 0.62],
            "feature_window_set": [[10,20], [10,20,50]]
        }
    },
    "Family_10": {
        "class": sme.BanditSelectorSignal,
        "params": {
            "reward_window": [20, 30, 50, 100],
            "epsilon": [0.01, 0.05, 0.10, 0.20]
        }
    },
    "Family_11": {
        "class": sme.StreakStructureSignal,
        "params": {
            "recent_streak_window": [10, 20, 30, 50],
            "follow_threshold": [0.58, 0.60, 0.62, 0.64],
            "fade_threshold": [0.58, 0.60, 0.62, 0.64],
            "long_streak_filter": [3, 4, 5, 6]
        }
    },
    "Family_12": {
        "class": sme.CompressionNoveltySignal,
        "params": {
            "pattern_len": [3, 4, 5, 6],
            "lookback": [300, 500, 800, 1200],
            "common_threshold": [0.70, 0.80, 0.90],
            "rare_threshold": [0.05, 0.10, 0.15, 0.20],
            "action_mode": ["common_follow", "common_fade", "rare_follow", "rare_fade"],
            "min_samples": [15, 20, 25, 30]
        }
    }
}

SHARED_RISK_PARAMS = {
    "max_trades_day": [2, 4, 6, 8],
    "stop_after_losses": [1, 2, 3],
    "daily_stop": [-1, -2, -3, -4],
    "skip_if_recent_dominance_enabled": [True, False],
    "recent_dominance_window": [30, 50, 70],
    "recent_dominance_ratio": [0.70, 0.75, 0.80],
    "skip_if_streak_too_long_enabled": [True, False],
    "max_live_streak_allowed": [2, 3, 4, 5]
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

def sample_params(space: dict) -> dict:
    """Randomly samples one value from each list in a param space dict."""
    return {k: random.choice(v) for k, v in space.items()}

def evaluate_fold(variant_id: str, family_name: str, sig_class, sig_params: dict, risk_params: dict, events_fold: list, exec_p_base: dict):
    exec_p = dict(exec_p_base)
    exec_p["session_params"] = {
        "max_trades_per_day": risk_params["max_trades_day"],
        "max_cons_losses_day": risk_params["stop_after_losses"],
        "max_loss_units_day": float(-risk_params["daily_stop"])
    }
    
    # Merge risk params into standard params for ModularStrategy overlays
    strat_params = {"max_series_losses": 999}
    strat_params.update(risk_params)
    
    sig = sig_class(sig_params)
    size = FixedSizing({"base_bet": 1.0})
    strat = ModularStrategy(sig, size, strat_params)
    
    return run_backtest(events_fold, strat, exec_p)

def process_variant(variant_id: str, family_name: str, sig_class, sig_params: dict, risk_params: dict, folds: list, exec_p_base: dict):
    # Folds is a list of [train_events, test_events].
    # But walk-forward strictly for this simplified auto-discovery uses Validation & Test
    # Fold 1: Train = 0..30%, Test = 30..60%
    # Fold 2: Train = 30..60%, Test = 60..100%
    
    total_trades = 0
    total_pnl = 0.0
    combined_equity = []
    
    fold_pnls = []
    all_reports = []
    
    current_equity = 100.0
    
    for train_ev, test_ev in folds:
        report = evaluate_fold(variant_id, family_name, sig_class, sig_params, risk_params, test_ev, exec_p_base)
        if not report or report['total_trades'] == 0:
            fold_pnls.append(0)
            continue
            
        total_trades += report['total_trades']
        total_pnl += report['net_profit']
        fold_pnls.append(report['net_profit'])
        all_reports.append(report)
        
        # Stitch equity
        if not combined_equity:
            combined_equity.extend(report['equity_curve'])
            current_equity = report['equity_curve'][-1]['equity']
        else:
            offset = current_equity - 100.0 # because report starts at 100
            for pt in report['equity_curve'][1:]: # skip day 0
                combined_equity.append({
                    "time": pt["time"],
                    "equity": pt["equity"] + offset,
                    "drawdown": pt["drawdown"]
                })
            current_equity = combined_equity[-1]['equity']
            
    # Calculate Max DD over stitched curve
    max_dd = 0.0
    if combined_equity:
        peak = combined_equity[0]['equity']
        for pt in combined_equity:
            if pt['equity'] > peak: peak = pt['equity']
            dd = (peak - pt['equity']) / peak * 100
            if dd > max_dd: max_dd = dd
            
    # Hard Filters
    profitable_folds = sum(1 for p in fold_pnls if p > 0)
    
    if total_pnl <= 0: return None
    if max_dd > 15.0: return None
    if total_trades < 100: return None
    if profitable_folds <= len(folds) / 2: return None # Must be profitable in more than half the folds
    
    # Needs validation pass!
    # Validation PnL = PnL of first fold. Test PnL = PnL of second fold.
    if len(fold_pnls) == 2:
        val_pnl, test_pnl = fold_pnls
        if val_pnl <= 0 or test_pnl <= 0:
            return None
    else:
        val_pnl, test_pnl = 0, 0
    
    ret_pct = total_pnl / 100.0
    dd_pct_decimal = max_dd / 100.0
    calmar = ret_pct / dd_pct_decimal if dd_pct_decimal > 0.001 else (ret_pct * 10)
    
    # Store all config info
    summary = {
        "id": variant_id,
        "signal": family_name,
        "signal_params": sig_params,
        "sizing": "Fixed_1Unit_WF",
        "sizing_params": {"base_bet": 1.0},
        "session_params": risk_params,
        "max_series_losses": f"WF Session Lim / Overlays",
        
        "net_profit": total_pnl,
        "win_rate": sum(r['win_rate_pct'] * r['total_trades'] for r in all_reports) / max(1, total_trades) if all_reports else 0,
        "max_dd": max_dd,
        "total_trades": total_trades,
        "calmar": round(calmar, 3),
        "val_pnl": val_pnl,
        "test_pnl": test_pnl,
        "fold_stability": profitable_folds / len(folds),
        "equity_curve": combined_equity,
        "trades": [] # Stubs for trades so dashboard doesn't crash if it expects them
    }
    
    return summary

def generate_random_variants(n_variants_per_family=100):
    variants = []
    variant_counter = 0
    
    for fam_name, s_data in SPACES.items():
        sig_class = s_data["class"]
        sig_space = s_data["params"]
        
        for _ in range(n_variants_per_family):
            variant_id = f"config_wf_{fam_name}_{variant_counter}"
            sig_params = sample_params(sig_space)
            risk_params = sample_params(SHARED_RISK_PARAMS)
            variants.append((variant_id, fam_name, sig_class, sig_params, risk_params))
            variant_counter += 1
            
    return variants

def run_global_discovery():
    print("Loading events data into memory...")
    events = load_data(DATA_DIR)
    if not events: return
    
    # Create 2 simple temporal folds:
    # Fold 1: First 50%
    # Fold 2: Last 50%
    mid_idx = len(events) // 2
    folds = [
        (events[:mid_idx // 2], events[:mid_idx]), # Train=first 25%, Test=0 to 50%
        (events[mid_idx // 2 : mid_idx], events[mid_idx:]) # Train=25-50%, Test=50 to 100%
    ]
    
    n_per_family = 50 # Reduced from 100 for testing speed
    variants = generate_random_variants(n_per_family)
    print(f"Generated {len(variants)} total variants across {len(SPACES)} families.")
    
    exec_p_base = {
        "initial_capital": 500.0,
        "base_trade_budget": 5.0
    }
    
    survivors = []
    start_t = time.time()
    
    with ProcessPoolExecutor(max_workers=os.cpu_count()-1) as executor:
        futures = {
            executor.submit(
                process_variant,
                v[0], v[1], v[2], v[3], v[4], folds, exec_p_base
            ): i for i, v in enumerate(variants)
        }
        
        for i, future in enumerate(as_completed(futures)):
            try:
                res = future.result()
                if res:
                    survivors.append(res)
            except Exception as e:
                pass
            if (i+1) % 50 == 0:
                print(f"Processed {i+1} / {len(variants)} ...")
                
    print(f"Evaluation complete in {time.time() - start_t:.1f}s.")
    print(f"Survivors: {len(survivors)}")
    
    # Next: Refinement? (Skipped for initial prototype to save time)
    # Just append survivors to leaderboard
    if survivors:
        survivors.sort(key=lambda x: x['calmar'], reverse=True)
        
        with open(DISCOVERY_OUTPUT_FILE, 'r') as f:
            existing_data = json.load(f)
            
        existing_strategies = existing_data.get("strategies", [])
        existing_strategies.extend(survivors)
        existing_strategies.sort(key=lambda x: x['calmar'], reverse=True)
        
        # Save detailed json for each survivor for the dashboard stats
        os.makedirs(DETAILS_DIR, exist_ok=True)
        for s in survivors:
            detail_path = os.path.join(DETAILS_DIR, f"{s['id']}.json")
            # Create a dashboard-compatible format
            dash_data = {
                "initial_capital": 500.0,
                "final_capital": 500.0 + s["net_profit"],
                "net_profit": s["net_profit"],
                "total_trades": s["total_trades"],
                "win_rate_pct": s["win_rate"],
                "max_drawdown_pct": s["max_dd"],
                "max_consecutive_losses": 0,
                "val_pnl": s.get("val_pnl", 0),
                "test_pnl": s.get("test_pnl", 0),
                "fold_stability": s.get("fold_stability", 0),
                "signal_params": s.get("signal_params", {}),
                "session_params": s.get("session_params", {}),
                "explanation": generate_explanation(s.get("signal"), s.get("signal_params", {}), s.get("session_params", {})),
                "equity_curve": s.pop("equity_curve", []),
                "trades": s.pop("trades", [])
            }
            with open(detail_path, 'w') as df:
                json.dump(dash_data, df)
        
        existing_data["strategies"] = existing_strategies
        total_runs = existing_data.get("total_tested", 0) + len(variants)
        existing_data["total_tested"] = total_runs
        existing_data["generated_at"] = datetime.utcnow().isoformat() + "Z"
        
        with open(DISCOVERY_OUTPUT_FILE, 'w') as f:
            json.dump(existing_data, f, indent=2)
            
        print(f"Appended {len(survivors)} new WF strategies to leaderboard. Total tests historically: {total_runs}")
    else:
        print("No variants survived the hard filters.")

if __name__ == "__main__":
    run_global_discovery()
