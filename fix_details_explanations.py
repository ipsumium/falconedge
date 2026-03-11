import json
import os

BASE_DIR = "/Users/zmeura/Documents/polymarket/FalconEdge"

def fix_details():
    with open(os.path.join(BASE_DIR, "discovery_results.json"), "r") as f:
        main_data = json.load(f)
        
    strats = {}
    for s in main_data.get("strategies", []):
        strats[s["id"]] = s
        
    details_dir = os.path.join(BASE_DIR, "discovery_details")
    for f in os.listdir(details_dir):
        if not f.endswith(".json"): continue
        path = os.path.join(details_dir, f)
        base_id = f.replace(".json", "")
        
        if base_id.startswith("live_"):
            parts = base_id.split("_", 2)
            if len(parts) == 3:
                base_id = parts[2]
                
        if base_id in strats:
            with open(path, "r") as dfile:
                try:
                    detail = json.load(dfile)
                except:
                    continue
                    
            detail["explanation"] = strats[base_id].get("explanation", "")
            
            # If it's a live file, it won't have a summary, but let's inject missing keys at root!
            if "summary" not in detail:
                 if "signal" not in detail: detail["signal"] = strats[base_id].get("signal", "")
                 if "sizing" not in detail: detail["sizing"] = strats[base_id].get("sizing", "")
                 if "sizing_params" not in detail: detail["sizing_params"] = strats[base_id].get("sizing_params", {})
            else:
                 detail["summary"]["explanation"] = detail["explanation"]
                
            with open(path, "w") as dfile:
                json.dump(detail, dfile)
                
    print("Injected explanation, signal, and sizing into discovery_details!")

if __name__ == "__main__":
    fix_details()
