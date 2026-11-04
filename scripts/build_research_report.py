"""Build readable Markdown reports from the published aggregate results.

This needs only the Python standard library: no credentials or raw data.
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'reports/oos_2026'
NAMES = {
    'garch_t': 'GARCH-t', 'ewma': 'EWMA', 'rolling21': 'Rolling21',
    'har_proxy_ridge': 'HAR proxy / Ridge',
    'ml_hist_gradient_boosting': 'Histogram gradient boosting',
}


def read(name, folder=DATA):
    with (folder / name).open(newline='') as source:
        return list(csv.DictReader(source))


def money(value):
    n = float(value)
    return ('−' if n < 0 else '+') + f'{abs(n):,.0f}'


def number(value, digits=2):
    return f'{float(value):,.{digits}f}'


def integer(value):
    return str(int(float(value)))


def table(headers, rows):
    return '\n'.join([
        '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join('---' for _ in headers) + ' |',
        *['| ' + ' | '.join(map(str, row)) + ' |' for row in rows],
    ])


def main():
    summary = read('summary.csv')
    daily = [r for r in summary if r['protocol_id'] == 'daily_close']
    hourly = [r for r in summary if r['protocol_id'] == 'elapsed_60m']
    scores = sorted(read('forecast_scores.csv'), key=lambda r: float(r['qlike']))
    by_horizon = read('forecast_scores_by_horizon.csv')
    costs = [r for r in read('cost_breakdown.csv') if r['protocol_id'] == 'daily_close']
    development = read('longer_hedge_comparison.csv', ROOT / 'reports/development_2022_2025')

    def trade_rows(records):
        return [[NAMES[r['model']], integer(r['completed']), money(r['before_execution_pnl']),
                 money(r['after_execution_pnl']),
                 integer(r['before_wins']) + ' / ' + integer(r['after_wins']),
                 integer(r['vol_bets_correct_15m']) + ' / ' + integer(r['vol_bets_scorable'])]
                for r in records]

    trade_table = table(['Model', 'Trades', 'Before execution costs ($)', 'After execution costs ($)',
                         'Profitable before / after', 'RV > entry ask-IV'], trade_rows(daily))
    hourly_table = table(['Model', 'Trades', 'Before execution costs ($)', 'After execution costs ($)',
                          'Profitable before / after', 'RV > entry ask-IV'], trade_rows(hourly))
    forecast_table = table(['Model', 'Forecasts', 'QLIKE ↓', 'MAE (vol points) ↓', 'RMSE (vol points) ↓'],
        [[NAMES[r['model']], integer(r['observations']), number(r['qlike'],6),
          number(r['mae_vol_points'],3), number(r['rmse_vol_points'],3)] for r in scores])
    horizon_table = table(['Forecast horizon (sessions)', 'Model', 'QLIKE ↓', 'MAE (vol points) ↓'],
        [[integer(r['horizon_sessions']), NAMES[r['model']], number(r['qlike'],4), number(r['mae_vol_points'],2)]
         for r in sorted(by_horizon, key=lambda r:(int(r['horizon_sessions']),float(r['qlike'])))])
    cost_table = table(['Model', 'Option executed spread ($)', 'Stock spread ($)', 'Slippage ($)',
                        'Option fees ($)', 'Stock fees ($)', 'Total execution cost ($)'],
        [[NAMES[r['model']], *[number(r[k],2) for k in ['option_executed_spread_cost',
           'stock_spread_cost','slippage_cost','option_fees','stock_fees','total_execution_cost']]] for r in costs])
    risk_table = table(['Model', 'Mean trade return before (%)', 'Mean trade return after (%)',
                        'All-opportunity mean return after (%)', 'Mean absolute residual Delta (shares)'],
        [[NAMES[r['model']], number(100*float(r['mean_return_on_capital_before']),4),
          number(100*float(r['mean_return_on_capital_after']),4),
          number(100*float(r['all_opportunity_mean_roc_after']),4),
          number(r['mean_abs_delta_shares'],2)] for r in daily])
    development_table = table(['Hedge schedule', 'Trades', 'Before execution costs ($)',
                              'After execution costs ($)', 'Mean absolute residual Delta (shares)'],
        [[r['protocol_id'],integer(r['trades']),money(r['no_execution_cost']),money(r['observed_costs']),
          number(r['mean_abs_delta_shares'],2)] for r in development])

    report = f'''# Volatility Forecasting and Delta-Hedged Single Options

## Research finding

Five volatility models were evaluated on a common monthly opportunity grid for 20 US stocks. Forecast variance was matched to each option's remaining life. The policy bought a single call or put when forecast volatility exceeded that contract's ask-implied volatility by more than three absolute volatility percentage points, then delta-hedged with stock.

In the January–July 2026 entry cohorts, **all five models lost money after execution costs under both daily-close and hourly hedging**. GARCH and histogram gradient boosting had nearly identical aggregate QLIKE, but gradient boosting selected a worse-performing trade subset. There is no demonstrated durable trading edge or universal model winner from this sample.

![Forecast accuracy and daily-hedge trading results](images/oos_model_comparison.png)

## 1. Sample and research chronology

| Item | Definition |
| --- | --- |
| Development period | 2022–2025; outcomes already inspected during prior research |
| New historical time holdout | First NYSE trading session of each month, January–July 2026 |
| Entry dates | January 2 through July 1, 2026 |
| Actual exits | January 8 through September 17, 2026 |
| Target tenors | 7, 30 and 90 calendar days; actual eligible expiries used |
| Opportunity count | 20 stocks × 7 monthly origins × 3 tenors = 420 per model |
| Primary hedge schedule | Daily, at exchange close minus 15 minutes |
| Fixed comparison schedule | Every 60 elapsed minutes, with orders restricted to regular sessions |
| Entry signal | Forecast vol − decision ask-IV > 0.03 |
| Selection size | One standard option contract, multiplier 100, per selected stock/month/tenor |

The universe is AAPL, MSFT, NVDA, AMZN, JPM, KO, AMD, INTC, ADBE, CRM, QCOM, XOM, CVX, BAC, WFC, PEP, PG, UNH, MRK and DIS. It is a research convenience universe selected in earlier work; it is not a point-in-time reconstruction of an investable stock index.

The five models generated 442 selected entries altogether. Because selections overlap, these correspond to **249 distinct contract opportunities**. Two hedge schedules and two cost books give **996 unique replay scenarios**. There were no rejected entries in this run.

The January–July cutoff was specified to allow all admissible long-tenor contracts to finish within the available history; origins were not included or excluded according to realized profits.

### What “out of sample” means here

The strategy rules, parameters, threshold and two hedge schedules were fixed before this study's 2026 option-chain acquisition and P&L calculation. The 2022–2025 exploratory results were already known. Daily stock history through 2026 had already been cached, so this is a **historical chronological holdout**, not a claim that all 2026 prices had never been viewed, and not a prospective live trial.

Models refit at each monthly origin. Features use the prior actual session close. Supervised rows enter training only when the complete forward-return label block has finished by that cutoff. A 90-day-type label that is still unfolding cannot enter training merely because its anchor date is in the past. Earlier, completed test-period labels may enter later fits: this is walk-forward OOS rather than a model trained once and frozen for the entire year.

Publishing these results makes subsequent use of the same dates a reproduction or further exploration, not another untouched test.

## 2. Forecast models

| Model | Specification |
| --- | --- |
| GARCH-t | Zero-mean Student-t GARCH(1,1), fitted on at most 756 past daily returns |
| EWMA | Squared-return variance with decay 0.94 |
| Rolling21 | Mean of the latest 21 squared returns |
| HAR proxy / Ridge | Pooled regularized regression with daily variance proxies and fixed alpha 10 |
| Histogram gradient boosting | Pooled positive-variance model with Poisson loss; 150 iterations, learning rate 0.05, maximum depth 4, fixed random seed |

The two supervised models pool the 20 stocks, using a rolling 756-anchor training window. Future variance is modeled in blocks covering sessions 1, 2–5, 6–21, 22–63 and 64–84. Feature inputs include past returns, variance at multiple horizons, downside variance, high/low ranges, overnight gaps and relative volume. No option IV is supplied as a model feature. Hyperparameters are fixed rather than selected on holdout P&L.

Daily variance curves are integrated to the **actual option expiry**, using actual exchange sessions and ACT/365 annualization. Partial close-to-close intervals allocate variance in proportion to elapsed wall-clock time, including weekends. This is an approximation; intraday volatility seasonality is not estimated.

The HAR model is explicitly a **daily-proxy version**, not the original high-frequency HAR-RV specification. Labels are future squared daily close-to-close log returns; they are not a continuous realized-variance measure.

## 3. Single-option trading policy

1. On the first exchange session of each month, observe the stock and option chain at 09:45 New York time.
2. For each target tenor, choose the nearest eligible expiry. Tenor tolerances are 5, 12 and 25 calendar days for 7-, 30- and 90-day targets.
3. Find the nearest valid strike separately for the call and put; invert each ask price using the American-option approximation.
4. Choose the individual contract with the largest forecast-vol-minus-ask-IV gap, with a deterministic call tie-break. Buy only if that gap exceeds 0.03. Otherwise remain flat.
5. Enter at 10:00 New York time using valid observed quotes. Do not switch to another leg after a failed entry.
6. Hedge whole shares using one-minute-old **market-IV Delta**. Initialize the hedge independently of the later periodic schedule.
7. Plan to exit 15 minutes before the official close on the session before expiry, subject to quote eligibility and the fixed research capital-reserve rule. Unwind remaining stock at exit.

Within a given stock/month/tenor, a model's forecast is common to the call and put comparison. Maximizing forecast minus ask-IV therefore selects the lower-ask-IV valid candidate. Models primarily differ in whether that winning contract crosses their entry threshold. If several models choose the same opportunity, the exact same market-IV hedge book is reused; forecast values do not otherwise alter its hedge orders or exit rules.

This study does not enter paired straddles. Every model scans all 420 opportunities rather than inheriting an earlier fixed trade list.

## 4. Forecast accuracy

Forecast accuracy is scored independently of whether an option trade is entered. Each model has 420 scored forecasts: 20 stocks × 7 origins × 5/21/63-session horizons. These scoring horizons are separate from the option-trading tenors, whose forecasts are integrated to their actual expiries.

For realized integrated variance `v` and forecast integrated variance `v_hat`, the normalized QLIKE used here is:

```text
r = v / v_hat
QLIKE = r - log(r) - 1
```

Lower QLIKE is better. MAE and RMSE below are annualized volatility percentage-point errors. The realized target is the daily squared-return proxy described above.

{forecast_table}

Gradient boosting and GARCH are effectively tied on aggregate QLIKE; GARCH has slightly smaller aggregate MAE and RMSE. Rankings vary by horizon: gradient boosting has the lowest 5-session QLIKE, while the HAR proxy has the lowest 21- and 63-session QLIKE. Different forecast-loss rankings should not be read as trading-profit rankings.

{horizon_table}

## 5. Trading results

### Daily-close hedging: primary comparison

Dollar amounts are sums of one-contract, independently funded trades. Each model starts with the same 420 opportunities; counts differ because their forecasts trigger different entry decisions.

{trade_table}

“RV > entry ask-IV” compares actual entry-to-exit 15-minute-plus-overnight realized volatility with the option's **actual entry-time ask-IV**. It is a volatility-bet diagnostic, not a profit count or a direct measure of forecast error. Gamma weights, IV repricing, discrete hedges, holding-window differences and costs can make it disagree with P&L.

### Hourly hedging: fixed comparison

{hourly_table}

All five models remain negative after execution costs. Gradient boosting's daily-hedge book is already negative before execution costs; its hourly pre-cost result is approximately flat. The four other models have positive pre-execution-cost totals but negative observed-cost totals.

### Capital normalization and residual risk

Each trade is independently funded at three times its entry 100-share underlying notional. The all-opportunity mean includes flat or failed-entry opportunities as zero and averages over all 420 opportunities. These are diagnostic cohort averages, not annualized returns or the performance of a combined capital-constrained portfolio.

{risk_table}

Daily hedging leaves materially more residual Delta than hourly hedging. For example, GARCH's average absolute residual model Delta is about 9.96 shares under daily-close hedging versus 2.43 under hourly hedging. Lower transaction cost therefore comes with different exposure to the realized price path.

### Exit exception

The INTC February 2, 2026 90-day-target call triggered the fixed research capital-reserve close on May 8, ahead of its planned May 14 exit. EWMA, Rolling21 and HAR selected this opportunity; GARCH and gradient boosting did not. Daily and hourly triggers differed by one minute. Each midpoint comparison follows its corresponding actual exit. This reserve is a research constraint, not a simulated broker-margin requirement. The other 248 distinct opportunities exited at their planned times.

## 6. Execution costs and cash accounting

| Component | Fixed assumption |
| --- | --- |
| Option execution | Buy at observed ask, sell at observed bid |
| Option commission | $1 per contract per executed transaction |
| Stock execution | Buy ask / sell bid, plus 1 bp extra slippage |
| Stock commission | $0.0035 per share; $0.35 minimum per order |
| Stock sell surcharge | 0.1 bp of sale proceeds |
| Cash lending | max(SOFR − 0.5%, 0) |
| Cash borrowing | SOFR + 1.5% |
| Short-stock borrow | 2% annualized |
| Signal timing | At least one-minute-old stock/Delta information |
| Trading hours | Exchange regular sessions only; no overnight hedge orders |
| Sizing | Whole shares; observed sizes, with a stock quote-size multiplier of 100 |

Rate observations use a two-calendar-day availability lag. Cash dividends are booked separately. An unbid long right can be abandoned for zero recovery under the specified eligibility rule; the engine does not invent an executable midpoint sale.

The **before-execution-cost** book keeps the observed book's order times, quantities and exits, substitutes midpoint prices for executed orders, and removes commissions, sell surcharges and slippage. It still retains funding, short-stock borrow, the idle-cash benchmark and abandoned-rights writeoffs. Removing fees changes subsequent cash interest slightly, so the before/after P&L difference need not exactly equal the immediate execution-cost sum.

Daily-close execution-cost attribution:

{cost_table}

Option executed-spread costs exclude abandonment writeoffs. Writeoffs and financing are separate fields in the released cost CSV, avoiding double counting.

The cash bridge is:

```text
net P&L above idle-cash benchmark
  = option midpoint mark change + stock hedge mark P&L + stock dividends
    - execution spreads - slippage - commissions and sell surcharges
    - abandoned-rights writeoffs
    + cash interest - benchmark interest - stock borrow
```

## 7. Development results: context, not a second untouched test

The earlier 2022–2025 monthly GARCH-only single-option study scanned 2,880 opportunities and executed 990 trades. A fixed-entry hedge-schedule comparison produced the following totals:

{development_table}

The daily-close result was close to break-even before the later holdout but remained negative after costs. These outcomes had already been inspected. The 2026 primary daily-close and hourly comparison was fixed subsequently, before the new option P&L evaluation. These development results are not independent confirmation of the 2026 findings.

Calendar-hour schedules count nights and weekends but place orders only during regular sessions. A 12-calendar-hour interval therefore often results in an adjustment near the next session open. Cumulative trading-hour schedules exclude nights, weekends, holidays and early-close downtime.

## 8. Validation and limitations

The original complete local run passed 55 unit tests, including future-price perturbation checks that leave past forecasts unchanged, supervised-label completion/purging tests, expiry variance integration, hedge clocks, and accounting tests. Its independent audit checked **102,570 fill records across observed and midpoint books**, with maximum cash-transition residual below $0.000000001 and zero recorded fill-price/fee discrepancies.

The public repository preserves the numerical sources, fixed plans, aggregate tables and original audit manifests. It excludes raw market data, full per-minute books, fitted models and credentials. The public release check reconciles available aggregates and their provenance; it does not claim to repeat the full historical-quote audit without the original local inputs.

Material limits include seven monthly entry cohorts, overlapping long tenors, cross-stock dependence, a convenience stock universe, independently funded rather than netted portfolio accounts, fixed fee/borrow assumptions, unverified underlying quote-tick freshness, an American-pricing approximation with continuous dividend yield, and no actual early-exercise simulation. Missing IV can be carried forward from past observations for at most four days; missing Delta observations remain disclosed. Discrete regular-session hedging leaves overnight and intraday gap risk.

These results do not establish a profitable strategy. They show how expiry-matched forecasting, signal selection, financing and actual execution assumptions can produce different conclusions from forecast accuracy alone.

## Files

- [All model/hedge summaries](../reports/oos_2026/summary.csv)
- [Forecast scores](../reports/oos_2026/forecast_scores.csv)
- [Monthly trading breakdown](../reports/oos_2026/monthly_breakdown.csv)
- [Stock trading breakdown](../reports/oos_2026/stock_breakdown.csv)
- [Tenor trading breakdown](../reports/oos_2026/tenor_breakdown.csv)
- [Execution-cost attribution](../reports/oos_2026/cost_breakdown.csv)
- [Original complete-run audit manifest](../reports/oos_2026/verification.json)
- [Frozen OOS plan](../oos_single_leg_plan.json)
- [Reproduction guide](REPRODUCIBILITY.md)
'''
    chinese = f'''# 波动率预测与单腿期权 Delta 对冲：研究报告

## 研究结论

在 20 只美股、2026 年 1–7 月每月入场的历史时间留出测试中，**五个模型在每日和每小时对冲下，扣除执行成本后均亏损**。GARCH 与梯度提升树的整体 QLIKE 几乎相同，但两者选出的期权交易表现明显不同。因此，这个样本没有证明某个模型能稳定赚钱，也不能仅凭整体预测评分选择交易模型。

![五模型预测与交易比较](images/oos_model_comparison.png)

## 测试区间与样本外规则

- **2022–2025 年：** 已用于研究模型与策略规则，不能重新当作未见过的测试集。
- **2026 年 1–7 月：** 本轮新增历史时间留出。首次入场 1 月 2 日，最后入场 7 月 1 日；实际退出延伸至 9 月 17 日。
- **共同机会：** 20 股 × 7 个月 × 7/30/90 天目标期限 = 每个模型 420 个机会；未触发信号就空仓。
- **滚动训练：** 每月使用前一交易日收盘及以前的信息。监督训练标签必须已经完整走完，尚未结束的长期标签不能使用。测试期早先月份已完成的标签可以加入后续月份训练。
- 模型参数、3 个绝对波动率百分点门槛及每日收盘前／每小时两种对冲安排，均在本轮 2026 期权链获取与损益计算前固定。

这属于 **walk-forward out-of-sample**，不是年底训练一次之后全年不更新。此前已经缓存了 2026 年日线，股票池和策略选择也受早期研究影响；因此它是历史时间留出，不是前瞻实盘验证。公开结果后再使用相同日期，属于复现或继续探索。

## 策略

每月首个交易日纽约 09:45 决策，10:00 入场。每个股票／月份／目标期限，先选最近的合格到期日，再分别找最近有效行权价的 Call 与 Put，比较“匹配实际剩余期限的预测波动率 − 各自 ask-IV”。优势最大的一张超过 3 个百分点才买入，最多一张标准 100 股乘数合约。

买入后使用市场 IV 推导的 Delta、至少一分钟前的信号和整股股票对冲；只在正常交易时段成交。计划在到期前一个交易日收盘前 15 分钟退出。这里没有买入跨式，也没有沿用早期固定的 22 笔交易名单。

同一个机会中，预测值对 Call、Put 相同，胜出合约由有效候选中较低的 ask-IV 决定。各模型主要在是否触发门槛上不同。同一机会被多个模型选中时，共用相同的真实报价、对冲和现金账本。五个模型合计选择 442 次，涉及 249 个不同合约机会；两种对冲和两种成本账本共 996 个独立回放场景，没有入场失败。

## 五个模型

GARCH-t 为零均值、Student-t 创新的 GARCH(1,1)；EWMA 衰减率为 0.94；Rolling21 使用最近 21 个平方日收益。HAR proxy/Ridge 与梯度提升树在 20 股上合并训练，以过去 756 个锚点为窗口，预测未来 1、2–5、6–21、22–63、64–84 个交易日区块方差。超参数固定，不使用期权 IV 作为模型特征。

预测的是平方日收益代理方差，HAR 为代理版本，并非标准高频 HAR-RV。每日预测方差积分至实际到期日，以 ACT/365 年化；部分收盘间隔按自然时间分摊，尚未估计日内季节性。

## 预测评分

每个模型在全部股票与月份的 5/21/63 交易日预测上评分，共 420 条，不以是否入场为条件。QLIKE 使用实际方差与预测方差之比 r，计算 r − log(r) − 1，越低越好。MAE/RMSE 的单位为波动率百分点。

{forecast_table}

梯度提升树与 GARCH 的整体 QLIKE 基本打平；GARCH 的 MAE/RMSE 略低。分期限看，一周左右的 QLIKE 是梯度提升树最好，一个月和三个月是 HAR proxy 较好。评分口径与期限不同，排名也会不同。

## 交易结果：每日收盘前 15 分钟对冲

下表金额为每笔一张期权、独立资金账户的美元损益加总。

{trade_table}

“RV > entry ask-IV”是实际持有期 15 分钟加隔夜 RV 高于实际入场 ask-IV 的次数，不是盈利次数。Gamma 权重、IV 重定价、离散对冲及成本都会使两者不同。

## 每小时对冲的对照

{hourly_table}

五个模型扣费后仍全部亏损。梯度提升树的每日对冲结果在不扣执行成本时已亏损，而每小时对冲的不扣执行成本结果接近零；其余四个模型的不扣执行成本总额为正，扣费后转负。

## 交易成本究竟是什么

| 成本 | 假设 |
| --- | --- |
| 期权价差 | 买 ask、卖 bid |
| 期权佣金 | 每张每次实际成交 $1 |
| 股票价差与滑点 | 买 ask、卖 bid，另加成交价 1bp 滑点 |
| 股票佣金 | 每股 $0.0035，每单最低 $0.35 |
| 股票卖出附加费 | 卖出金额的 0.1bp |
| 现金利息 | 贷出 max(SOFR−0.5%, 0)，借入 SOFR+1.5% |
| 股票借券 | 年化 2% |

“不扣执行成本”使用同一实际交易的时间、数量和退出时点，将可执行成交换成中间价，并去掉佣金、附加费及滑点；仍保留融资、借券、闲置现金利息基准和无人报价期权放弃的损失。移除费用也会略微改变后续现金利息，因此前后损益差额不必完全等于即时执行费用之和。

每日对冲执行成本拆分如下。期权已成交价差不含权利放弃损失，避免重复相加。

{cost_table}

## 资金、风险与提前退出

每笔资金基数为入场 100 股标的名义金额的三倍；这不是实际经纪商保证金。全部机会平均收益把空仓计为零；它和逐笔平均收益都不是年化收益或净额投资组合收益。

{risk_table}

每日对冲会留下更多方向暴露，例如 GARCH 的平均绝对残余 Delta 约 9.96 股，每小时约 2.43 股。

INTC 的 2026 年 2 月 2 日、90 天目标 Call 于 5 月 8 日触发既定研究资金储备规则，早于原计划 5 月 14 日退出。EWMA、Rolling21、HAR 选中了该机会；GARCH 与梯度提升树未选中。每日与每小时触发相差一分钟，中间价对照分别固定至实际退出。其余 248 个不同机会均按计划退出。

## 审计与结论边界

完整本地研究通过 55 项测试，包含改变未来价格后历史预测不变、监督标签完成时间、方差积分、对冲时点和现金账检查。独立审计核对了实际与中间价账本中的 102,570 条成交记录；最大现金变化残差低于十亿分之一美元，成交价与费用没有记录差异。

公开仓库保留数值源代码、固定计划、汇总结果及原审计记录，不包含原始行情、完整逐分钟账本、拟合模型或 API key。公开检查验证可见汇总及其来源，不声称在缺少原始输入时重做了完整行情审计。

七个月入场批次较短，长期交易重叠、股票相关，股票池存在选择偏差。独立账户没有跨交易资金或对冲净额；借券和费用固定，报价逐笔更新新鲜度无法独立验证，期权采用美式近似定价和连续股息率，没有模拟实际提前行权。缺失 IV 最多仅向前携带四天，离散对冲仍留下隔夜和跳跃风险。

本轮没有证明策略能稳定赚钱。较好的整体波动率预测评分，并不自动产生较好的交易信号或扣费后收益。

详细开发期结果、预测期限比较与方法见 [English report](RESEARCH_REPORT.md)。原始汇总见 [results](../reports/oos_2026/)，完整复现条件见 [reproduction guide](REPRODUCIBILITY.md)。
'''
    (ROOT/'docs/RESEARCH_REPORT.md').write_text(report,encoding='utf-8')
    (ROOT/'docs/RESEARCH_REPORT.zh-CN.md').write_text(chinese,encoding='utf-8')
    print('Built English and Chinese Markdown reports from published aggregate CSV files.')


if __name__ == '__main__':
    main()
