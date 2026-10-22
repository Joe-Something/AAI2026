Ad Budget Optimization Agent
An agent that splits a daily ad budget across Search, Social and Display to get the most conversions. It reads a performance CSV, computes CTR, CVR and CPA, moves 10–20% of the budget toward the best channel each day within hard guardrails, and logs a reason for every change.

How to use: Runtime → Run all (about 1 minute). No API key is needed. The Gemini policy in section 6 is optional.

Role and goal. Each day the agent splits a fixed daily budget ($3,000 in the demo) across Search, Social and Display to get the most conversions, which is the same as getting the lowest blended CPA. The goal can be switched to clicks with --objective clicks.

Inputs. A CSV with date, channel, spend, impressions, clicks, conversions: channel-level totals for 14–30 days. The config sets the daily budget, the guardrail limits and the objective.

Outputs. For each day, dollars per channel that add up to the budget. The agent also writes:

decision_log.csv: for each channel and day, the previous, proposed and final budget, the metrics, any guardrails that fired, and a reason. For example: "Social up 15.0% ( 1,000−> 1,150): best conversions per  (CVR3.2 28.46); +22% vs Search; stable CPA".
run_log.jsonl, a results summary and a chart.
