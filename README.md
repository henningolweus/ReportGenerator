# ReportGenerator

Automates creation of a PowerPoint investment factsheet by combining portfolio data, benchmark metrics, allocation breakdowns, custom charts, and LLM-powered commentary.

## TL;DR - Quick Start
- Install Python 3.11+ and activate a virtual environment.
- `pip install -r requirements.txt`
- Drop the latest Aggregated Amounts export into the folder referenced by `data_sources.aggregated_amounts_dir` (default `./input/aggregated/`); the newest matching file is picked up automatically.
- Update `config.yaml` with the new factsheet date (`as_of`) and any other settings that changed.
- Set `OPENAI_API_KEY` in your environment (`$env:OPENAI_API_KEY="sk-..."` on PowerShell) or add it to a `.env` file in the project root (auto-loaded via `python-dotenv`).
- (Optional) Inspect `processed_data/commentary_generation_cache.json` if you want to keep a running log of AI prompts/responses.
- Build the deck: `python -m factsheet.app build --config config.yaml`
- The PowerPoint lands in `dist/`. Close any open copy before rebuilding to avoid locks.

## Deep Dive

### 1. Pipeline Overview
1. **Configuration** – `config.yaml` defines report dates, data sources, output template, lookup tables, and commentary options.
2. **Data ingest (`factsheet.data`)** – Loads aggregated holdings and performance data from Excel, plus market data (via Yahoo Finance) to compute returns.
3. **Portfolio metrics (`factsheet.calc`)** – Calculates YTD, since-inception, CAGR, sector/region weights, and top holdings.
4. **Chart assembly (`factsheet.charts`)** – Builds line, bar, and allocation charts (with PNG fallbacks when native editing is unavailable).
5. **PPT templating (`factsheet.ppt`)** – Replaces shapes/tables/charts inside `Kukula Fact Sheet Growth 20250830.pptx` while preserving formatting.
6. **Commentary (`factsheet.commentary`)** – Optionally pulls market headlines via RSS, filters for macro vs. holdings relevance, and prompts OpenAI for a balanced narrative.
7. **Outputs** – The finished deck and supporting artefacts are written to `dist/` (e.g., latest PPT, classification review JSON, chart images).

Use the audit helper whenever you tweak the template:  
`python -m factsheet.app audit --config config.yaml`

### 2. Required Inputs & Artefacts
| File / Folder | Purpose |
| --- | --- |
| `input/aggregated/` (or whatever `aggregated_amounts_dir` points to) | Drop the latest Aggregated Amounts export here; the build automatically picks the newest matching file. |
| `Kukula Fact Sheet Growth 20250830.pptx` | PowerPoint template providing layout and placeholder names. |
| `input/mappings/instrument_overrides.csv`, `input/mappings/region_map.csv` | Map tickers/sectors to display buckets used in the pie charts. |
| `input/mappings/sector_aliases.csv` | Fallback sector labels for uncategorised holdings. |
| `fetched_data/yf_classification_cache.json` | Persistent cache for holdings classifications fetched from Yahoo Finance. |
| `processed_data/final_classification.json` | Generated after each run to review/edit resolved holdings. |
| `processed_data/commentary_generation_cache.json` | Latest commentary prompt/output (optional). |
| `fetched_data/news_cache.json` | Most recently fetched RSS headlines (optional). |

### 3. Key Sections in `config.yaml`
```yaml
as_of: 2025-09-30
start_date: 2021-09-06

data_sources:
  aggregated_amounts_dir: ./input/aggregated
  aggregated_amounts_pattern: AggregatedAmounts_*.xlsx

output:
  ppt_template: ./Kukula Fact Sheet Growth 20250830.pptx
  ppt_out_dir: ./dist
  charts_dir: ./dist/charts
```

- `tables.top_holdings_limit` caps the holdings table.
- `tables.instrument_overrides_file` / `region_map_file` point to the lookup CSVs.
- `data_sources.aggregated_amounts_dir` / `aggregated_amounts_pattern` control where the builder looks for the latest Aggregated Amounts export.
- `tables.classification_cache` is where ticker classifications persist between runs.
- `tables.classification_review_file` is the JSON generated for manual review each cycle.

#### Commentary Configuration
```yaml
commentary:
  enabled: true
  model: gpt-4o-mini
  temperature: 0.6
  mode: mixed               # options: all, macro, holdings, mixed
  max_per_feed: 3
  max_headlines: 4
  max_macro_headlines: 2
  max_holdings_headlines: 2
  rss_feeds:
    - https://www.reuters.com/finance/markets/rss
    - https://feeds.a.dj.com/rss/RSSMarketsMain.xml
    - https://www.ft.com/markets?format=rss
  exclude_keywords:
    - boeing
  history_file: ./processed_data/commentary_generation_cache.json
```
Add `include_keywords` to force certain tickers/themes through the filter, or set `enabled: false` to author the commentary manually.

### 4. Manual Overrides & Customisation
1. **Sector/region mapping** – Edit `input/mappings/instrument_overrides.csv` and `input/mappings/region_map.csv` to change how holdings roll up into display buckets.
2. **Fallback sector names** – Update `input/mappings/sector_aliases.csv` for any instruments that need friendlier labels.
3. **Holdings overrides** – Adjust `processed_data/final_classification.json` after a run (e.g., tweak `override_sector`), then rebuild. Changes stick via `fetched_data/yf_classification_cache.json`.
4. **Template updates** – Modify `Kukula Fact Sheet Growth 20250830.pptx`, ensure placeholders retain their names, then run the audit command to confirm they’re still detected.

### 5. Operational Tips
- Close any generated PPT before rebuilding; Windows locks the file while it’s open.
- Keep dependencies isolated in a virtual environment (`python -m venv .venv` then `.\.venv\Scripts\Activate.ps1`).
- Ensure the Aggregated Amounts export includes the expected "Aggregated Amounts" sheet and column names (Date, Amount Client Currency, etc.).
- When offline or skipping AI usage, set `commentary.enabled` to `false`.
- `processed_data/commentary_generation_cache.json` stores only the latest commentary output, while `fetched_data/news_cache.json` retains the newest headlines.

With the inputs in place and `config.yaml` updated, regenerating the deck is a single command each reporting period.
