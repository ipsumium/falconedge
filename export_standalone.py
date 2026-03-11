import os
import json
import zlib
import base64

DATA_DIR = "/Users/zmeura/Documents/polymarket/FalconEdge"
RESULTS_FILE = os.path.join(DATA_DIR, "discovery_results.json")
DETAILS_DIR = os.path.join(DATA_DIR, "discovery_details")
OUTPUT_HTML = os.path.join(DATA_DIR, "tinyhost_dashboard.html")

def build_standalone():
    print("Loading discovery_results.json...")
    with open(RESULTS_FILE, 'r') as f:
        main_data = json.load(f)
        
    print(f"Loaded {len(main_data['strategies'])} strategies.")
    
    # Filter to only keep top 500 profitable strategies by calmar
    profitable_strats = [s for s in main_data['strategies'] if s.get('calmar', 0) > 0]
    main_data['strategies'] = sorted(profitable_strats, key=lambda x: x.get('calmar', 0), reverse=True)[:500]
    
    # Compress the main strategies array
    # We will downsample the equity_curve to max 150 points per strategy to save MBs.
    for strat in main_data['strategies']:
        curve = strat.get('equity_curve', [])
        if len(curve) > 150:
            step = len(curve) / 150.0
            strat['equity_curve'] = [curve[int(i * step)] for i in range(150)]
            
    print("Loading detailed JSONs for the Top 500 strategies...")
    all_details = {}
    
    for strat in main_data['strategies']:
        cid = strat['id']
        detail_path = os.path.join(DETAILS_DIR, f"{cid}.json")
        if os.path.exists(detail_path):
            with open(detail_path, 'r') as f:
                det = json.load(f)
                
                # Keep max 500 recent trades
                trades = det.get('trades', [])
                if len(trades) > 500:
                    det['trades'] = trades[-500:]
                    
                # We don't need duplicate summary objects since it's already in main_data
                if 'summary' in det:
                    del det['summary']
                
                all_details[cid] = det
                
    # Create HTML structure with two main "views"
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FalconEdge Standalone Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { background-color: #0f172a; color: #f8fafc; font-family: 'Inter', sans-serif; }
        .card { background-color: #1e293b; border-color: #334155; }
        .positive { color: #22c55e; }
        .negative { color: #ef4444; }
        .row-expanded { background-color: #334155 !important; }
        th.sortable:hover { color: #fff; cursor: pointer; }
        th.sorted-asc::after { content: " \\2191"; }
        th.sorted-desc::after { content: " \\2193"; }
    </style>
</head>
<body class="p-6">
    <div id="view-leaderboard" class="max-w-7xl mx-auto">
        <div class="flex justify-between items-center mb-6">
            <div>
                <h1 class="text-3xl font-bold text-white tracking-tight">Strategy Discovery Leaderboard</h1>
                <p class="text-sm text-gray-400 mt-1">Generated: <span id="gen-date">--</span> | Configurations Tested: <span id="total-tested">--</span></p>
            </div>
            <button onclick="window.location.reload()" class="px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded text-white font-medium transition">
                Reset View
            </button>
        </div>

        <div class="card p-4 rounded-lg border mb-6 flex gap-4 text-sm items-center">
            <span class="font-bold text-gray-300">Filters:</span>
            <label class="flex items-center space-x-2 text-gray-400">
                <span>Min Trades:</span>
                <input type="number" id="filter-trades" value="0" class="w-16 bg-gray-800 border border-gray-600 rounded px-2 py-1 text-white">
            </label>
            <label class="flex items-center space-x-2 text-gray-400">
                <span>Rule/Signal:</span>
                <input type="text" id="filter-strategy" placeholder="e.g. Markov" class="w-32 bg-gray-800 border border-gray-600 rounded px-2 py-1 text-white placeholder-gray-500">
            </label>
            <label class="flex items-center space-x-2 text-gray-400">
                <span>Min Win Rate (%):</span>
                <input type="number" id="filter-winrate" value="0" class="w-16 bg-gray-800 border border-gray-600 rounded px-2 py-1 text-white">
            </label>
            <label class="flex items-center space-x-2 text-gray-400">
                <span>Max Drawdown (%):</span>
                <input type="number" id="filter-dd" value="100" class="w-16 bg-gray-800 border border-gray-600 rounded px-2 py-1 text-white">
            </label>
            <label class="flex items-center space-x-2 text-gray-400">
                <span>Min Profit ($):</span>
                <input type="number" id="filter-profit" value="-10000" class="w-20 bg-gray-800 border border-gray-600 rounded px-2 py-1 text-white">
            </label>
            <button onclick="applyFilters()" class="px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded text-white transition ml-auto">Apply</button>
        </div>

        <div class="card rounded-lg border overflow-hidden">
            <div class="overflow-x-auto max-h-[800px]">
                <table class="w-full text-sm text-left text-gray-300">
                    <thead class="text-xs uppercase bg-gray-800 text-gray-400 sticky top-0 z-10">
                        <tr>
                            <th scope="col" class="px-2 py-3 sortable w-10 whitespace-nowrap" data-sort="rank">Rank</th>
                            <th scope="col" class="px-2 py-3 w-1/3 min-w-[250px] max-w-[400px]">Signal (Param)</th>
                            <th scope="col" class="px-2 py-3 w-1/6">Sizing Rule</th>
                            <th scope="col" class="px-2 py-3 sortable whitespace-nowrap" data-sort="total_trades">Trades</th>
                            <th scope="col" class="px-2 py-3 sortable whitespace-nowrap" data-sort="win_rate">Win Rate</th>
                            <th scope="col" class="px-2 py-3 sortable whitespace-nowrap" data-sort="max_dd">Max DD</th>
                            <th scope="col" class="px-2 py-3 sortable whitespace-nowrap text-center" data-sort="max_series_losses">Max Streak</th>
                            <th scope="col" class="px-2 py-3 sortable whitespace-nowrap text-right" data-sort="net_profit">Net Profit ($-)</th>
                            <th scope="col" class="px-2 py-3 sortable whitespace-nowrap text-right sorted-desc text-white" data-sort="calmar">Risk-Adj Return</th>
                        </tr>
                    </thead>
                    <tbody id="table-body" class="divide-y divide-gray-700"></tbody>
                </table>
            </div>
        </div>
    </div>

    <!-- Hidden template for expanded row chart -->
    <template id="chart-row-template">
        <tr class="chart-row">
            <td colspan="8" class="p-4 bg-gray-900 border-b border-gray-700">
                <div class="flex justify-between items-center mb-4">
                    <span class="text-gray-300 text-sm font-medium">Quick Equity View</span>
                    <button class="detail-link px-4 py-1.5 bg-blue-600 hover:bg-blue-500 rounded text-white text-xs font-bold shadow transition">Open Full Stats Page</button>
                </div>
                <div class="grid grid-cols-2 gap-4">
                    <div class="bg-gray-800 p-4 rounded" style="height: 250px;"><canvas class="eq-chart"></canvas></div>
                    <div class="bg-gray-800 p-4 rounded" style="height: 250px;"><canvas class="dd-chart"></canvas></div>
                </div>
            </td>
        </tr>
    </template>

    <!-- FULL STATS VIEW (Hidden initially) -->
    <div id="view-stats" class="max-w-7xl mx-auto hidden">
        <div class="flex justify-between items-center mb-6">
            <h1 class="text-3xl font-bold text-white tracking-tight">FalconEdge Backtest Details</h1>
            <button onclick="showLeaderboard()" class="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded text-white font-medium transition">
                &larr; Back to Leaderboard
            </button>
        </div>

        <div class="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-4 mb-6">
            <div class="card p-4 rounded-lg border">
                <p class="text-sm text-gray-400 font-medium">Net Profit</p>
                <p id="kpi-pnl" class="text-2xl font-bold mt-1">--</p>
            </div>
            <div class="card p-4 rounded-lg border">
                <p class="text-sm text-gray-400 font-medium">Win Rate</p>
                <p id="kpi-winrate" class="text-2xl font-bold text-white mt-1">--%</p>
            </div>
            <div class="card p-4 rounded-lg border">
                <p class="text-sm text-gray-400 font-medium">Total Trades</p>
                <p id="kpi-trades" class="text-2xl font-bold text-white mt-1">--</p>
                <p class="text-xs text-gray-500 mt-1"><span id="kpi-wins" class="positive"></span> / <span id="kpi-losses" class="negative"></span></p>
            </div>
            <div class="card p-4 rounded-lg border">
                <p class="text-sm text-gray-400 font-medium">Max Drawdown</p>
                <p id="kpi-dd" class="text-2xl font-bold negative mt-1">--%</p>
            </div>
            <div class="card p-4 rounded-lg border">
                <p class="text-sm text-gray-400 font-medium">Max Losing Streak</p>
                <p id="kpi-streak" class="text-2xl font-bold text-orange-400 mt-1">--</p>
            </div>
            <div class="card p-4 rounded-lg border">
                <p class="text-sm text-gray-400 font-medium">Days Traded</p>
                <p id="kpi-days" class="text-2xl font-bold text-white mt-1">--</p>
            </div>
            <div class="card p-4 rounded-lg border">
                <p class="text-sm text-gray-400 font-medium">Avg Daily PnL</p>
                <p id="kpi-daily-pnl" class="text-2xl font-bold mt-1">--</p>
                <p class="text-xs text-gray-500 mt-1"><span id="kpi-daily-pct">--%</span> per day</p>
            </div>
        </div>

        <div id="explanation-box" class="card p-4 mb-6 rounded-lg border text-gray-300">
            <h3 class="text-lg font-bold text-white mb-2" id="expl-title">Strategy Description</h3>
            <p id="expl-text" class="whitespace-pre-wrap"></p>
        </div>

        <div class="card p-4 mb-6 rounded-lg border text-gray-400 text-sm flex justify-between items-center whitespace-nowrap overflow-x-auto gap-4">
            <div>Signal Params: <span id="param-sig" class="text-white font-medium">--</span></div>
            <div>Max Loss Streak Limit: <span id="param-limit" class="text-white font-medium">--</span></div>
            <div>Base Budget: <span id="param-base" class="text-white font-medium">--</span></div>
            <div>Starting Capital: <span id="param-cap" class="text-white font-medium">--</span></div>
        </div>

        <div class="grid grid-cols-1 gap-6 mb-6">
            <div class="card p-4 rounded-lg border" style="height: 350px;"><canvas id="equityChartFull"></canvas></div>
            <div class="card p-4 rounded-lg border" style="height: 200px;"><canvas id="drawdownChartFull"></canvas></div>
        </div>

        <div class="card rounded-lg border overflow-hidden">
            <div class="p-4 border-b border-gray-700 font-medium text-white"><h3>Trade Log (Last 500 Trades Max)</h3></div>
            <div class="overflow-x-auto max-h-[500px]">
                <table class="w-full text-sm text-left text-gray-300 whitespace-nowrap">
                    <thead class="text-xs uppercase bg-gray-800 text-gray-400 sticky top-0">
                        <tr>
                            <th scope="col" class="px-6 py-3">Time/Date (UTC)</th>
                            <th scope="col" class="px-6 py-3">Direction</th>
                            <th scope="col" class="px-6 py-3">Shares</th>
                            <th scope="col" class="px-6 py-3">Escalation Stage</th>
                            <th scope="col" class="px-6 py-3">Forced Exit (99c)</th>
                            <th scope="col" class="px-6 py-3 text-right">Net PnL</th>
                        </tr>
                    </thead>
                    <tbody id="trade-table-body" class="divide-y divide-gray-700"></tbody>
                </table>
            </div>
        </div>
    </div>

    <!-- Inject Compressed JSON Payload -->
    <script>
"""
    
    # We dump it as tightly as possible
    payload_str = json.dumps({
        'main': main_data,
        'details': all_details
    }, separators=(',', ':'))
    
    html_content += f"window.FALCON_PAYLOAD = {payload_str};\n"

    html_content += """
        let fullData = window.FALCON_PAYLOAD.main.strategies;
        let detailsData = window.FALCON_PAYLOAD.details;
        
        let filteredData = [];
        let currentSort = { col: 'calmar', asc: false };
        let activeCharts = {}; 

        function init() {
            document.getElementById('gen-date').textContent = new Date(window.FALCON_PAYLOAD.main.generated_at).toLocaleString();
            document.getElementById('total-tested').textContent = window.FALCON_PAYLOAD.main.total_tested;
            applyFilters();
        }

        function applyFilters() {
            const minTrades = parseInt(document.getElementById('filter-trades').value) || 0;
            const minWr = parseFloat(document.getElementById('filter-winrate').value) || 0;
            const maxDd = parseFloat(document.getElementById('filter-dd').value) || 100;
            const minProfit = parseFloat(document.getElementById('filter-profit').value) || -10000;
            const stratStr = document.getElementById('filter-strategy').value.toLowerCase().trim();
            
            filteredData = fullData.filter(s => {
                const matchesTrades = s.total_trades >= minTrades;
                const matchesWr = s.win_rate >= minWr;
                const matchesDd = s.max_dd <= maxDd;
                const matchesProfit = s.net_profit >= minProfit;
                const matchesStrat = stratStr === '' || 
                                     s.signal.toLowerCase().includes(stratStr) || 
                                     s.sizing.toLowerCase().includes(stratStr);
                
                return matchesTrades && matchesWr && matchesDd && matchesProfit && matchesStrat;
            });
            sortData();
        }

        function sortData() {
            filteredData.sort((a, b) => {
                let valA = a[currentSort.col];
                let valB = b[currentSort.col];
                if (currentSort.col === 'rank') { valA = a.calmar; valB = b.calmar; }
                if (valA < valB) return currentSort.asc ? -1 : 1;
                if (valA > valB) return currentSort.asc ? 1 : -1;
                return 0;
            });
            renderTable();
        }

        function toggleSort(col) {
            if (currentSort.col === col) { currentSort.asc = !currentSort.asc; } 
            else { currentSort.col = col; currentSort.asc = false; }
            document.querySelectorAll('th.sortable').forEach(th => {
                th.classList.remove('sorted-asc', 'sorted-desc', 'text-white');
                if (th.dataset.sort === col) th.classList.add(currentSort.asc ? 'sorted-asc' : 'sorted-desc', 'text-white');
            });
            sortData();
        }

        document.querySelectorAll('th.sortable').forEach(th => {
            th.addEventListener('click', () => toggleSort(th.dataset.sort));
        });

        function renderTable() {
            const tbody = document.getElementById('table-body');
            tbody.innerHTML = '';
            Object.values(activeCharts).forEach(c => c.destroy());
            activeCharts = {};

            filteredData.forEach((strat, index) => {
                const tr = document.createElement('tr');
                tr.className = "hover:bg-gray-800 transition cursor-pointer main-row";
                
                const sigP = Object.entries(strat.signal_params).map(([k, v]) => `${k}=${v}`).join(', ');
                const sigStr = `<div class="font-bold">${strat.signal}</div><div class="text-xs text-gray-500 leading-tight mt-0.5 whitespace-normal break-words">${sigP}</div>`;
                const sizeStr = `<div class="font-medium">${strat.sizing}</div><div class="text-[10px] text-gray-500 mt-0.5 whitespace-normal break-words">(Max L: ${strat.max_series_losses})</div>`;
                const pnlClass = strat.net_profit >= 0 ? 'positive font-medium' : 'negative font-medium';
                const calmarClass = strat.calmar > 0 ? 'positive font-bold' : 'negative font-bold';

                tr.innerHTML = `
                    <td class="px-2 py-3 text-gray-500 whitespace-nowrap align-top">#${index + 1}</td>
                    <td class="px-2 py-3 text-white align-top w-1/3 min-w-[250px] max-w-[400px]">${sigStr}</td>
                    <td class="px-2 py-3 align-top">${sizeStr}</td>
                    <td class="px-2 py-3 whitespace-nowrap align-top">
                        ${strat.total_trades}
                        <div class="text-[10px] mt-0.5"><span class="text-green-400">${strat.wins || 0}W</span> <span class="text-red-400">${strat.losses || 0}L</span></div>
                    </td>
                    <td class="px-2 py-3 whitespace-nowrap align-top ${strat.win_rate >= 50 ? 'text-green-400' : 'text-red-400'}">${strat.win_rate.toFixed(2)}%</td>
                    <td class="px-2 py-3 whitespace-nowrap text-red-300 align-top">${strat.max_dd.toFixed(1)}%</td>
                    <td class="px-2 py-3 text-center text-orange-400 font-bold whitespace-nowrap align-top">${strat.max_series_losses}</td>
                    <td class="px-2 py-3 text-right ${pnlClass} whitespace-nowrap align-top">$${strat.net_profit.toFixed(2)}</td>
                    <td class="px-2 py-3 text-right ${calmarClass} whitespace-nowrap align-top">${strat.calmar.toFixed(3)}</td>
                `;
                tr.addEventListener('click', () => toggleRowExpansion(tr, strat));
                tbody.appendChild(tr);
            });
        }

        function toggleRowExpansion(row, stratData) {
            const isExpanded = row.classList.contains('row-expanded');
            document.querySelectorAll('.main-row').forEach(r => r.classList.remove('row-expanded'));
            document.querySelectorAll('.chart-row').forEach(r => r.remove());
            if (isExpanded) return;

            row.classList.add('row-expanded');
            const clone = document.getElementById('chart-row-template').content.cloneNode(true);
            const chartRow = clone.querySelector('tr');
            row.parentNode.insertBefore(chartRow, row.nextSibling);

            // Bind full detail button
            chartRow.querySelector('.detail-link').onclick = () => loadFullStats(stratData.id);

            const labels = []; const eqPts = []; const ddPts = [];
            stratData.equity_curve.forEach(pt => { labels.push(''); eqPts.push(pt.equity); ddPts.push(-pt.drawdown); });

            Chart.defaults.color = '#94a3b8';
            activeCharts[`eq_${stratData.id}`] = new Chart(chartRow.querySelector('.eq-chart').getContext('2d'), {
                type: 'line',
                data: { labels, datasets: [{ label: 'Equity', data: eqPts, borderColor: '#3b82f6', backgroundColor: 'rgba(59, 130, 246, 0.1)', fill: true, pointRadius: 0, borderWidth: 2, tension: 0.1 }] },
                options: { responsive: true, maintainAspectRatio: false, scales: { x: { display: false } } }
            });
            activeCharts[`dd_${stratData.id}`] = new Chart(chartRow.querySelector('.dd-chart').getContext('2d'), {
                type: 'line',
                data: { labels, datasets: [{ label: 'Drawdown (%)', data: ddPts, borderColor: '#ef4444', backgroundColor: 'rgba(239, 68, 68, 0.2)', fill: true, pointRadius: 0, borderWidth: 1, stepped: true }] },
                options: { responsive: true, maintainAspectRatio: false, scales: { x: { display: false } } }
            });
        }

        // FULL STATS LOGIC
        let fullEqChart = null;
        let fullDdChart = null;

        function loadFullStats(configId) {
            // Swap view
            document.getElementById('view-leaderboard').classList.add('hidden');
            document.getElementById('view-stats').classList.remove('hidden');
            window.scrollTo(0, 0);

            const summary = fullData.find(s => s.id === configId);
            const details = detailsData[configId] || { trades: [], explanation: "Detailed trade logs are missing.", execution_params: {initial_capital: 500} };

            document.getElementById('expl-title').textContent = `Strategy: ${summary.signal} + ${summary.sizing}`;
            document.getElementById('expl-text').textContent = details.explanation;
            
            document.getElementById('param-sig').textContent = Object.entries(summary.signal_params).map(([k,v]) => `${k}=${v}`).join(', ');
            document.getElementById('param-limit').textContent = summary.max_series_losses;
            document.getElementById('param-base').textContent = `$${details.execution_params ? details.execution_params.base_trade_budget : 5.0}`;
            document.getElementById('param-cap').textContent = `$${details.execution_params ? details.execution_params.initial_capital : (details.initial_capital || 500)}`;

            let pnl = summary.net_profit;
            const pnlEl = document.getElementById('kpi-pnl');
            pnlEl.textContent = `$${pnl.toFixed(2)}`;
            pnlEl.className = pnl >= 0 ? 'text-2xl font-bold mt-1 positive' : 'text-2xl font-bold mt-1 negative';
            
            document.getElementById('kpi-winrate').textContent = `${Number(summary.win_rate).toFixed(2)}%`;
            document.getElementById('kpi-trades').textContent = summary.total_trades;
            
            // Allow exact wins/losses from structure, or fallback to approximate
            document.getElementById('kpi-wins').textContent = `${summary.wins !== undefined ? summary.wins : Math.round(summary.total_trades * (summary.win_rate / 100))}W`;
            document.getElementById('kpi-losses').textContent = `${summary.losses !== undefined ? summary.losses : Math.round(summary.total_trades * (1 - (summary.win_rate / 100)))}L`;
            
            document.getElementById('kpi-dd').textContent = `${summary.max_dd.toFixed(2)}%`;
            document.getElementById('kpi-streak').textContent = summary.max_series_losses;

            // Calculate active trading days
            let daysTraded = 0; let avgDailyPnl = 0; let avgDailyPct = 0;
            if (summary.equity_curve && summary.equity_curve.length > 0) {
                const firstD = new Date(summary.equity_curve[0].time);
                const lastD = new Date(summary.equity_curve[summary.equity_curve.length-1].time);
                daysTraded = Math.ceil(Math.abs(lastD - firstD) / (1000 * 60 * 60 * 24)) || 1;
                avgDailyPnl = pnl / daysTraded;
                let initCap = details.execution_params ? details.execution_params.initial_capital : (details.initial_capital || 500);
                avgDailyPct = (avgDailyPnl / initCap) * 100;
            }
            document.getElementById('kpi-days').textContent = daysTraded;
            
            const pnlDailyEl = document.getElementById('kpi-daily-pnl');
            pnlDailyEl.textContent = `$${avgDailyPnl.toFixed(2)}`;
            pnlDailyEl.className = avgDailyPnl >= 0 ? 'text-2xl font-bold mt-1 positive' : 'text-2xl font-bold mt-1 negative';
            const pctD = document.getElementById('kpi-daily-pct');
            pctD.textContent = `${avgDailyPct >= 0 ? '+' : ''}${avgDailyPct.toFixed(2)}%`;
            pctD.className = avgDailyPct >= 0 ? 'positive' : 'negative';

            // Charts
            const chartCurve = details.equity_curve && details.equity_curve.length > 0 ? details.equity_curve : summary.equity_curve;
            const labels = []; const eqPts = []; const ddPts = [];
            chartCurve.forEach(pt => {
                const date = new Date(pt.time);
                labels.push(`${date.getMonth() + 1}/${date.getDate()} ${date.getHours().toString().padStart(2, '0')}:${date.getMinutes().toString().padStart(2, '0')}`);
                eqPts.push(pt.equity);
                ddPts.push(-pt.drawdown);
            });

            if(fullEqChart) fullEqChart.destroy();
            fullEqChart = new Chart(document.getElementById('equityChartFull').getContext('2d'), {
                type: 'line',
                data: { labels, datasets: [{ label: 'Portfolio Equity ($)', data: eqPts, borderColor: '#3b82f6', backgroundColor: 'rgba(59, 130, 246, 0.1)', fill: true, tension: 0.1, pointRadius: 0 }] },
                options: { responsive: true, maintainAspectRatio: false, interaction: { intersect: false, mode: 'index' }, scales: { x: { display: false } } }
            });

            if(fullDdChart) fullDdChart.destroy();
            fullDdChart = new Chart(document.getElementById('drawdownChartFull').getContext('2d'), {
                type: 'line',
                data: { labels, datasets: [{ label: 'Drawdown (%)', data: ddPts, borderColor: '#ef4444', backgroundColor: 'rgba(239, 68, 68, 0.2)', fill: true, tension: 0, stepped: true, pointRadius: 0 }] },
                options: { responsive: true, maintainAspectRatio: false, interaction: { intersect: false, mode: 'index' }, scales: { x: { display: false } } }
            });

            const tbody = document.getElementById('trade-table-body');
            tbody.innerHTML = '';
            details.trades.slice().reverse().forEach((trade) => {
                const tr = document.createElement('tr');
                tr.className = "hover:bg-gray-800 transition";
                const pnlClass = trade.pnl >= 0 ? 'positive font-medium' : 'negative font-medium';
                tr.innerHTML = `
                    <td class="px-6 py-3 text-gray-400">${new Date(trade.time).toLocaleString()}</td>
                    <td class="px-6 py-3 uppercase font-bold text-${trade.direction === 'up' ? 'green' : 'red'}-500">${trade.direction}</td>
                    <td class="px-6 py-3">${trade.size_shares.toFixed(2)}</td>
                    <td class="px-6 py-3">Step ${trade.series_step}</td>
                    <td class="px-6 py-3">${trade.forced_exit ? '<span class="px-2 py-1 bg-blue-900/50 text-blue-400 rounded text-xs border border-blue-800">FILLED 99c</span>' : '-'}</td>
                    <td class="px-6 py-3 text-right ${pnlClass}">${trade.pnl >= 0 ? '+' : ''}$${trade.pnl.toFixed(2)}</td>
                `;
                tbody.appendChild(tr);
            });
        }

        function showLeaderboard() {
            document.getElementById('view-stats').classList.add('hidden');
            document.getElementById('view-leaderboard').classList.remove('hidden');
        }

        window.addEventListener('DOMContentLoaded', init);
    </script>
</body>
</html>
"""
    
    with open(OUTPUT_HTML, "w", encoding='utf8') as f:
        f.write(html_content)
        
    size_mb = os.path.getsize(OUTPUT_HTML) / (1024 * 1024)
    print(f"Generated standalone dashboard {OUTPUT_HTML} (Size: {size_mb:.2f} MB)")

if __name__ == "__main__":
    build_standalone()
