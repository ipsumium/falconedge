import json
import os

BASE_DIR = "/Users/zmeura/Documents/polymarket/FalconEdge"

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

def fix():
    for filename in ["discovery_results.json", "live_results_2026-03-07.json", "live_results_2026-03-08.json", "live_results_2026-03-09.json"]:
        path = os.path.join(BASE_DIR, filename)
        if not os.path.exists(path):
            continue
        with open(path, "r") as f:
            data = json.load(f)
        for s in data.get("strategies", []):
            s["explanation"] = generate_full_explanation(s)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Fixed {filename}")

    # Also fix details
    details_dir = os.path.join(BASE_DIR, "discovery_details")
    if os.path.exists(details_dir):
        for f in os.listdir(details_dir):
            if f.endswith(".json"):
                path = os.path.join(details_dir, f)
                with open(path, "r") as dfile:
                    try:
                        detail = json.load(dfile)
                        detail["explanation"] = generate_full_explanation(detail.get("summary", detail))
                        if "summary" in detail:
                            detail["summary"]["explanation"] = detail["explanation"]
                    except:
                        continue
                with open(path, "w") as dfile:
                    json.dump(detail, dfile)
    print("Fixed details JSONs.")

if __name__ == "__main__":
    fix()
