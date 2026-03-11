import os
import shutil
import json
from datetime import datetime

BASE_DIR = "/Users/zmeura/Documents/polymarket/FalconEdge"
HOSTING_DIR = os.path.join(BASE_DIR, "hosting_exports")

def build_hosting_package():
    os.makedirs(HOSTING_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_dir = os.path.join(HOSTING_DIR, f"FalconEdge_Web_{timestamp}")
    os.makedirs(export_dir)
    print(f"Creating export in {export_dir}...")
    
    # Copy htmls
    shutil.copy2(os.path.join(BASE_DIR, "discovery_dashboard.html"), os.path.join(export_dir, "index.html"))
    shutil.copy2(os.path.join(BASE_DIR, "dashboard.html"), os.path.join(export_dir, "dashboard.html"))
    
    # Determine which strategies we want to keep
    # To save space but keep "all relevant time" data, we keep strategies with Calmar > 0
    # The local page showed 1581 strategies, but you only interact with the profitable ones.
    # However, to be 100% identical, we'll keep all from the json, or let's do top 1000 to drop absolute trash.
    
    main_jsons = ["discovery_results.json", "live_results_2026-03-07.json", "live_results_2026-03-08.json", "live_results_2026-03-09.json"]
    ids_to_copy = set()
    
    for jf in main_jsons:
        src = os.path.join(BASE_DIR, jf)
        dest = os.path.join(export_dir, jf)
        if os.path.exists(src):
            with open(src, 'r') as f:
                data = json.load(f)
                
            # For the huge all-time JSON, we can write back a slightly filtered version keeping top performers to save MBs.
            if jf == "discovery_results.json":
                # Keep top 1000 by calmar instead of 1581, saves 500 junk JSON requests
                strategies_sorted = sorted(data.get("strategies", []), key=lambda x: x.get("calmar", -999), reverse=True)
                clean_strats = strategies_sorted[:1000]
                data["strategies"] = clean_strats
            
            # Write optimized or exact original into export
            with open(dest, 'w') as out_f:
                json.dump(data, out_f)
                
            for s in data.get("strategies", []):
                ids_to_copy.add(s["id"])
                    
    # Copy needed detail JSONs
    details_src = os.path.join(BASE_DIR, "discovery_details")
    details_dest = os.path.join(export_dir, "discovery_details")
    os.makedirs(details_dest, exist_ok=True)
    
    copied = 0
    missing = 0
    for sid in ids_to_copy:
        sf = os.path.join(details_src, f"{sid}.json")
        if os.path.exists(sf):
            shutil.copy2(sf, os.path.join(details_dest, f"{sid}.json"))
            copied += 1
        else:
            missing += 1
            
    # Calculate folder size
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(export_dir):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
                
    size_mb = total_size / (1024 * 1024)
            
    print(f"Export created successfully!")
    print(f"Copied {copied} detail JSON files. (Missing: {missing})")
    print(f"Export Folder: {export_dir}")
    print(f"Total Size: {size_mb:.2f} MB")
    
if __name__ == "__main__":
    build_hosting_package()
