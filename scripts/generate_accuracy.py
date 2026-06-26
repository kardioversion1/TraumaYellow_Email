#!/usr/bin/env python3
"""
generate_accuracy.py
Reads data/predictions_history.csv + data/signals.csv
Writes docs/accuracy.json (served via GitHub Pages)
Run: python scripts/generate_accuracy.py
"""
import csv, json, os, sys
from datetime import datetime, timedelta
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED_PATH = os.path.join(BASE, 'data', 'predictions_history.csv')
SIG_PATH  = os.path.join(BASE, 'data', 'signals.csv')
OUT_PATH  = os.path.join(BASE, 'docs', 'accuracy.json')

def sf(v, default=0):
    try: return float(v) if v and str(v).strip() else default
    except: return default

# ── Load & join ──────────────────────────────────────────────────────────────
pred_rows = list(csv.DictReader(open(PRED_PATH)))
sig_rows  = list(csv.DictReader(open(SIG_PATH))) if os.path.exists(SIG_PATH) else []
sig_by_date = {r['date']: r for r in sig_rows}

data = []
for r in pred_rows:
    try:
        sig = sig_by_date.get(r['date'], {})
        act = r.get('actual','')
        if not act or not act.strip(): continue  # skip future rows
        data.append({
            'date':       r['date'],
            'pred':       float(r['predicted']),
            'lo':         float(r['band_lo']),
            'hi':         float(r['band_hi']),
            'act':        float(act),
            'flu_pct':    sf(sig.get('nssp_flu_pct')),
            'event_flag': sf(sig.get('event_attendance')) > 0,
            'snowfall':   sf(sig.get('snowfall_mm')),
            'is_holiday': sig.get('is_holiday','0') == '1',
        })
    except Exception as e:
        print(f"  skip {r.get('date','?')}: {e}", file=sys.stderr)

if not data:
    print("ERROR: no rows parsed", file=sys.stderr)
    sys.exit(1)

print(f"Parsed {len(data)} rows ({data[0]['date']} → {data[-1]['date']})")

# ── Surge threshold (75th pct of actuals) ────────────────────────────────────
acts_sorted = sorted(r['act'] for r in data)
p75 = acts_sorted[int(0.75 * len(acts_sorted))]

# ── Condition tagging ─────────────────────────────────────────────────────────
def tag(r):
    tags = []
    if r['flu_pct'] >= 3:   tags.append('flu')
    if r['event_flag']:      tags.append('event')
    if r['snowfall'] >= 3:  tags.append('snow/ice')
    if r['is_holiday']:     tags.append('holiday')
    return tags if tags else ['baseline']

# ── Helper stats ──────────────────────────────────────────────────────────────
def stats_for(sub):
    if not sub: return None
    n   = len(sub)
    mae = sum(abs(r['act']-r['pred']) for r in sub) / n
    ib  = sum(1 for r in sub if r['lo'] <= r['act'] <= r['hi'])
    w5  = sum(1 for r in sub if abs(r['act']-r['pred']) <= 5)
    sg  = [r for r in sub if r['act'] >= p75]
    sg_c = [r for r in sg  if r['pred'] >= p75 - 5]
    return {
        'n':             n,
        'mae':           round(mae, 2),
        'in_band_pct':   round(100*ib/n, 1),
        'within5_pct':   round(100*w5/n, 1),
        'surge_caught':  len(sg_c),
        'surge_total':   len(sg),
        'missed_surges': len(sg) - len(sg_c),
    }

# ── Time windows ──────────────────────────────────────────────────────────────
now_dt = datetime.fromisoformat(data[-1]['date'])
windows = {}
for days, key in [(14,'14d'),(30,'30d'),(60,'60d'),(90,'90d')]:
    cutoff = now_dt - timedelta(days=days)
    sub = [r for r in data if datetime.fromisoformat(r['date']) >= cutoff]
    if sub: windows[key] = stats_for(sub)

# ── Condition breakdown ───────────────────────────────────────────────────────
cond_stats = {}
for cond_key, cond_label in [
    ('baseline','Baseline (no flags)'),
    ('flu',     'Flu A elevated'),
    ('event',   'Event days'),
    ('snow/ice','Snow / ice days'),
    ('holiday', 'Holiday'),
]:
    sub = [r for r in data if cond_key in tag(r)]
    if len(sub) >= 3:
        s = stats_for(sub)
        cond_stats[cond_label] = {'mae': s['mae'], 'in_band_pct': s['in_band_pct'], 'n': s['n']}

multi = [r for r in data if len(tag(r)) > 1]
if len(multi) >= 3:
    s = stats_for(multi)
    cond_stats['Multi-flag days'] = {'mae': s['mae'], 'in_band_pct': s['in_band_pct'], 'n': s['n']}

# ── Day-of-week MAE ───────────────────────────────────────────────────────────
dow_err = defaultdict(list)
for r in data:
    dow = datetime.fromisoformat(r['date']).strftime('%a')
    dow_err[dow].append(abs(r['act'] - r['pred']))
dow_mae = {d: round(sum(v)/len(v), 2) for d,v in dow_err.items()}

# ── Notable misses ────────────────────────────────────────────────────────────
misses = sorted(
    [r for r in data if abs(r['act']-r['pred']) > 10],
    key=lambda x: abs(x['act']-x['pred']), reverse=True
)[:8]

# ── Chart series (full history) ───────────────────────────────────────────────
chart = [
    {'date': r['date'], 'pred': r['pred'], 'lo': r['lo'], 'hi': r['hi'], 'act': r['act']}
    for r in data
]

# ── Assemble ──────────────────────────────────────────────────────────────────
overall = stats_for(data)
accuracy = {
    'generated':    datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
    'data_through': data[-1]['date'],
    'note':         'computed from predictions_history.csv vs ed_counts.csv — real data',
    'total_rows':   len(data),
    'date_range':   [data[0]['date'], data[-1]['date']],
    'surge_threshold': p75,
    'overall':      overall,
    'windows':      windows,
    'condition_breakdown': cond_stats,
    'dow_mae':      dow_mae,
    'top_misses':   [
        {'date': m['date'], 'pred': round(m['pred']), 'lo': round(m['lo']),
         'hi': round(m['hi']), 'act': round(m['act']), 'delta': round(m['act']-m['pred'])}
        for m in misses
    ],
    'chart': chart,
}

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, 'w') as f:
    json.dump(accuracy, f, separators=(',', ':'))

kb = os.path.getsize(OUT_PATH) / 1024
print(f"Wrote {OUT_PATH} ({kb:.0f} KB)")
print(f"  MAE={overall['mae']}, within5={overall['within5_pct']}%, "
      f"surges caught={overall['surge_caught']}/{overall['surge_total']}")
