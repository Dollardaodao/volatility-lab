"""Offline consistency checks for the public research release.

Checks published sources and aggregates, not excluded historical quote books.
Uses only the Python standard library.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'reports/oos_2026'
MODELS={'garch_t','ewma','rolling21','har_proxy_ridge','ml_hist_gradient_boosting'}
PROTOCOLS={'daily_close','elapsed_60m'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    with (DATA/name).open(newline='') as f:return list(csv.DictReader(f))


def near(a,b,tolerance=1e-6):
    assert math.isfinite(float(a)) and math.isfinite(float(b))
    assert abs(float(a)-float(b))<=tolerance,(a,b)


def main():
    config=json.loads((DATA/'configuration.json').read_text())
    lock=json.loads((DATA/'protocol_lock.json').read_text())
    audit=json.loads((DATA/'verification.json').read_text())
    assert config['plan']==json.loads((ROOT/'oos_single_leg_plan.json').read_text())
    assert sha(ROOT/'oos_single_leg_plan.json')==lock['plan_sha256']
    assert sha(ROOT/'short_vol_plan.json')==config['cost_plan_sha256']
    for filename,expected in config['source_hashes'].items():
        assert sha(ROOT/'volatility_lab'/filename)==expected,filename
    for key,filename in [('report_source_sha256','oos_single_leg_report.py'),
                         ('audit_source_sha256','oos_single_leg_validate.py')]:
        assert sha(ROOT/'volatility_lab'/filename)==audit[key],filename
    assert sha(DATA/'summary.csv')==audit['primary_summary_sha256']
    assert audit['status']=='passed' and audit['unit_tests']==55
    assert audit['distinct_selected_cohorts']==249 and audit['scenario_rows']==996
    assert audit['entry_rejected_scenario_rows']==0
    assert audit['fill_records_checked']==102570
    assert audit['max_independent_cash_residual']<1e-5

    summary=read('summary.csv')
    keys={(r['model'],r['protocol_id']) for r in summary}
    assert len(summary)==10 and keys=={(m,p) for m in MODELS for p in PROTOCOLS}
    costs={(r['model'],r['protocol_id']):r for r in read('cost_breakdown.csv')}
    components=['option_executed_spread_cost','stock_spread_cost','slippage_cost','option_fees','stock_fees']
    for row in summary:
        n=int(row['completed']);assert int(row['opportunities'])==420
        assert n==int(row['signals']) and int(row['entry_rejected'])==0
        assert 0<=int(row['after_wins'])<=n and 0<=int(row['before_wins'])<=n
        assert int(row['calls'])+int(row['puts'])==n
        assert int(row['vol_bets_scorable'])==n
        assert 0<=int(row['vol_bets_correct_15m'])<=n
        near(row['vol_bet_hit_rate_15m'],int(row['vol_bets_correct_15m'])/n)
        near(row['all_opportunity_mean_roc_after'],float(row['mean_return_on_capital_after'])*n/420)
        cost=costs[row['model'],row['protocol_id']]
        near(sum(float(cost[c]) for c in components),cost['total_execution_cost'])
        near(cost['total_execution_cost'],row['execution_cost'])
        expected=float(row['market_pnl'])-float(row['execution_cost'])-float(cost['abandonment_writeoff'])+float(row['net_financing'])
        near(row['after_execution_pnl'],expected)
        assert float(row['after_execution_pnl'])<0
    assert sum(int(r['signals']) for r in summary if r['protocol_id']=='daily_close')==442

    # Each breakdown partitions the same completed books and opportunity grid.
    for filename in ['monthly_breakdown.csv','stock_breakdown.csv','tenor_breakdown.csv','kind_breakdown.csv']:
        records=read(filename)
        for row in summary:
            group=[r for r in records if (r['model'],r['protocol_id'])==(row['model'],row['protocol_id'])]
            assert sum(int(r['trades']) for r in group)==int(row['completed']),filename
            for field in ['before_execution_pnl','after_execution_pnl']:
                near(sum(float(r[field]) for r in group),row[field])
            if filename!='kind_breakdown.csv':assert sum(int(r['opportunities']) for r in group)==420

    scores=read('forecast_scores.csv')
    assert len(scores)==5 and {r['model'] for r in scores}==MODELS
    horizons=read('forecast_scores_by_horizon.csv');months=read('forecast_scores_by_month.csv')
    assert len(horizons)==15 and len(months)==35
    for row in scores:
        assert int(float(row['observations']))==420
        assert float(row['qlike'])>=0 and float(row['mae_vol_points'])>=0
        for records,count in [(horizons,3),(months,7)]:
            group=[r for r in records if r['model']==row['model']]
            assert len(group)==count and sum(int(float(r['observations'])) for r in group)==420
            for field in ['qlike','mae_vol_points']:
                near(sum(float(r[field])*float(r['observations']) for r in group)/420,row[field])

    # All committed Markdown links to local files must resolve.
    docs=[ROOT/'README.md',*(ROOT/'docs').glob('*.md')]
    for path in docs:
        for match in re.finditer(r'!?\[[^\]]*\]\(([^)\s]+)\)',path.read_text()):
            target=match.group(1)
            if target.startswith(('https://','http://','mailto:','#')):continue
            target=target.split('#',1)[0]
            assert (path.parent/target).exists(),(str(path.relative_to(ROOT)),target)

    manifest=json.loads((ROOT/'reports/publication_manifest.json').read_text())
    for filename,expected in manifest['files'].items():assert sha(ROOT/filename)==expected,filename
    print('PASS: frozen numerical sources/plans; 10 model/hedge summaries; all cost and P&L bridges;')
    print('      stock/month/tenor/leg partitions; five-model forecast coverage; Markdown links; release hashes.')
    print('Scope: published-source and aggregate checks only. Original complete quote/fill audit remains local.')


if __name__=='__main__':main()
