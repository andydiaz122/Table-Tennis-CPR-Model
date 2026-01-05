"""
Performance Validation Script
Compares current pipeline output against baseline metrics.
Returns exit code 0 if within tolerance, 1 if regression detected.
"""

import json
import sys
import os
from datetime import datetime

# Tolerance thresholds
ROI_MIN = 2.40  # Baseline 2.44%, allow -0.04%
SHARPE_MIN = 2.60  # Baseline 2.80, allow -0.20
MAX_DRAWDOWN_MAX = 35.00  # Baseline 34.12%, allow +0.88%

BASELINE_PATH = "logs/performance_snapshots/baseline.json"
BACKTEST_LOG = "V9.0/logs/08_backtest_final.log"


def parse_backtest_log(log_path):
    """Extract metrics from backtest log file."""
    metrics = {}

    if not os.path.exists(log_path):
        print(f"ERROR: Backtest log not found: {log_path}")
        return None

    with open(log_path, 'r') as f:
        for line in f:
            if "Return on Investment (ROI):" in line:
                metrics['roi'] = float(line.split(':')[1].strip().replace('%', ''))
            elif "Annualized Sharpe Ratio:" in line:
                metrics['sharpe_ratio'] = float(line.split(':')[1].strip())
            elif "Maximum Drawdown:" in line:
                metrics['max_drawdown'] = float(line.split(':')[1].strip().replace('%', ''))
            elif "Total Bets Placed:" in line:
                metrics['total_bets'] = int(line.split(':')[1].strip())
            elif "Total Profit:" in line:
                metrics['total_profit'] = float(line.split('$')[1].strip())

    return metrics


def load_baseline():
    """Load baseline metrics from JSON."""
    if not os.path.exists(BASELINE_PATH):
        print(f"ERROR: Baseline not found: {BASELINE_PATH}")
        return None

    with open(BASELINE_PATH, 'r') as f:
        return json.load(f)


def validate_metrics(current, baseline):
    """Compare current metrics against baseline with tolerances."""
    results = {
        'passed': True,
        'checks': [],
        'timestamp': datetime.now().isoformat()
    }

    # ROI Check
    roi_ok = current.get('roi', 0) >= ROI_MIN
    results['checks'].append({
        'metric': 'ROI',
        'current': current.get('roi'),
        'baseline': baseline['metrics']['roi'],
        'threshold': f">= {ROI_MIN}%",
        'passed': roi_ok
    })
    if not roi_ok:
        results['passed'] = False

    # Sharpe Check
    sharpe_ok = current.get('sharpe_ratio', 0) >= SHARPE_MIN
    results['checks'].append({
        'metric': 'Sharpe Ratio',
        'current': current.get('sharpe_ratio'),
        'baseline': baseline['metrics']['sharpe_ratio'],
        'threshold': f">= {SHARPE_MIN}",
        'passed': sharpe_ok
    })
    if not sharpe_ok:
        results['passed'] = False

    # Max Drawdown Check
    dd_ok = current.get('max_drawdown', 100) <= MAX_DRAWDOWN_MAX
    results['checks'].append({
        'metric': 'Max Drawdown',
        'current': current.get('max_drawdown'),
        'baseline': baseline['metrics']['max_drawdown'],
        'threshold': f"<= {MAX_DRAWDOWN_MAX}%",
        'passed': dd_ok
    })
    if not dd_ok:
        results['passed'] = False

    return results


def main():
    print("=" * 60)
    print("PERFORMANCE VALIDATION")
    print("=" * 60)

    # Load baseline
    baseline = load_baseline()
    if not baseline:
        sys.exit(1)

    print(f"\nBaseline: {baseline['commit']} ({baseline['branch']})")
    print(f"  ROI: {baseline['metrics']['roi']}%")
    print(f"  Sharpe: {baseline['metrics']['sharpe_ratio']}")
    print(f"  MaxDD: {baseline['metrics']['max_drawdown']}%")

    # Parse current metrics
    current = parse_backtest_log(BACKTEST_LOG)
    if not current:
        sys.exit(1)

    print(f"\nCurrent:")
    print(f"  ROI: {current.get('roi')}%")
    print(f"  Sharpe: {current.get('sharpe_ratio')}")
    print(f"  MaxDD: {current.get('max_drawdown')}%")

    # Validate
    results = validate_metrics(current, baseline)

    print(f"\n{'=' * 60}")
    print("VALIDATION RESULTS")
    print("=" * 60)

    for check in results['checks']:
        status = "PASS" if check['passed'] else "FAIL"
        print(f"  [{status}] {check['metric']}: {check['current']} (threshold: {check['threshold']})")

    print("=" * 60)

    if results['passed']:
        print("OVERALL: PASSED - Performance within acceptable tolerance")
        sys.exit(0)
    else:
        print("OVERALL: FAILED - Performance regression detected!")
        sys.exit(1)


if __name__ == "__main__":
    main()
