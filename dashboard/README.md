# Dashboard

```bash
pip install streamlit
streamlit run dashboard/app.py
```

Shows the bench results for the demo agent (or any saved report in
`results/*_report.json`): aggregate metrics (including swap
sensitivity), per-case detail, counterfactual results, citation-swap
probe results, and category breakdowns.

The app file is a thin presentation layer; all shaping logic lives in
`dashboard/helpers.py` (pure Python, no streamlit import) and is
covered by `tests/test_dashboard_helpers.py`.
