# ReportGenerator

Automates creation of a PowerPoint investment factsheet by merging portfolio inputs, Yahoo Finance market data, custom charts, and LLM-powered commentary.

## TL;DR - Quick Start
- Install Python 3.11+ and activate a virtual environment.
- `pip install -r requirements.txt`
- Drop the latest Aggregated Amounts export into `data/1_input_data/1_raw_financials/`. The newest `AggregatedAmounts_*.xlsx` is loaded automatically.
- Keep mapping CSVs under `data/1_input_data/2_mappings/` and the PPT template under `data/1_input_data/3_ppt_template/`.
- Update `config.yaml` with the new factsheet date (`as_of`) and any other settings that changed.
- Set `OPENAI_API_KEY` in your environment (`$env:OPENAI_API_KEY="sk-..."` on PowerShell) or add it to a `.env` file in the project root (auto-loaded via `python-dotenv`).
- Build the deck: `python -m factsheet.app build --config config.yaml`
- The builder combines raw inputs with fetched Yahoo Finance data cached in `data/2_fetched_data/`, writes processed artefacts to `data/3_processed_data/`, and saves the finished PPT in `dist/`.
- Close any generated PPT before rebuilding to avoid Windows file locks.

## Deep Dive

### 1. Pipeline Overview
1. **Configuration** - `config.yaml` defines report dates, file locations, output template, lookup tables, and commentary settings.
2. **Input prep (`data/1_input_data`)** - Raw financials, mapping tables, and the PPT template are loaded from their respective subfolders.
3. **Data ingest (`factsheet.data`)** - Reads the latest Aggregated Amounts workbook, fetches Yahoo Finance classifications and pricing (stored in `data/2_fetched_data/`), and merges everything into a unified dataset.
4. **Portfolio metrics (`factsheet.calc`)** - Calculates performance metrics, allocation splits, and top holdings based on the combined dataset.
5. **Chart assembly (`factsheet.charts`)** - Builds line, bar, and allocation charts, exporting PNG fallbacks if needed.
6. **Commentary (`factsheet.commentary`)** - Pulls RSS headlines, selects macro vs. holdings stories, and prompts OpenAI; prompts/responses are cached in `data/3_processed_data/`.
7. **PPT templating (`factsheet.ppt`)** - Injects tables, charts, and commentary into the PPT template while preserving formatting.
8. **Outputs** - The finished deck and supporting artefacts (classification review JSON, commentary history, chart images) land in `dist/` and `data/3_processed_data/`.

Use the audit helper whenever you tweak the template:  
`python -m factsheet.app audit --config config.yaml`

### 2. Required Inputs and Key Artefacts
| File / Folder | Purpose |
| --- | --- |
| `data/1_input_data/1_raw_financials/` | Drop the latest Aggregated Amounts export here; the build picks the newest `AggregatedAmounts_*.xlsx`. |
| `data/1_input_data/2_mappings/instrument_overrides.csv`, `region_map.csv`, `sector_aliases.csv` | Map tickers/sectors to the display buckets used in charts and tables. |
| `data/1_input_data/3_ppt_template/Kukula Fact Sheet Growth 20250830.pptx` | PowerPoint template providing layout and placeholder names. |
| `data/2_fetched_data/yf_classification_cache.json` | Persistent cache for holdings classifications fetched from Yahoo Finance. |
| `data/2_fetched_data/news_cache.json` | Latest RSS headlines fetched for commentary (optional). |
| `data/3_processed_data/final_classification.json` | Generated after each run to review/edit resolved holdings; feeds back into the cache. |
| `data/3_processed_data/commentary_generation_cache.json` | Stores the latest commentary prompt/output (optional). |

### 3. Key Sections in `config.yaml`
```yaml
as_of: 2025-09-30
start_date: 2021-09-06

data_sources:
  aggregated_amounts_dir: ./data/1_input_data/1_raw_financials
  aggregated_amounts_pattern: AggregatedAmounts_*.xlsx
  instrument_overrides_file: ./data/1_input_data/2_mappings/instrument_overrides.csv
  region_map_file: ./data/1_input_data/2_mappings/region_map.csv
  sector_aliases_file: ./data/1_input_data/2_mappings/sector_aliases.csv
  classification_cache_file: ./data/2_fetched_data/yf_classification_cache.json
  news_cache_file: ./data/2_fetched_data/news_cache.json
  commentary_history_file: ./data/3_processed_data/commentary_generation_cache.json

output:
  ppt_template: ./data/1_input_data/3_ppt_template/Kukula Fact Sheet Growth 20250830.pptx
  ppt_out_dir: ./dist
  charts_dir: ./dist/charts
  classification_review_file: ./data/3_processed_data/final_classification.json
```

- `tables.top_holdings_limit` caps the holdings table.
- The `data_sources` paths describe how raw inputs and fetched Yahoo Finance data are combined into the processed dataset.
- `output.classification_review_file` is regenerated each cycle for manual review and feeds back into the persisted cache.

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
  history_file: ./data/3_processed_data/commentary_generation_cache.json
```
Add `include_keywords` to force certain tickers/themes through the filter, or set `enabled: false` to craft the commentary manually.

### 4. Manual Overrides and Customisation
1. **Sector and region mapping** - Edit the CSVs in `data/1_input_data/2_mappings/` to change how holdings roll up into display buckets.
2. **Fallback sector names** - Update `sector_aliases.csv` for any instruments that need friendlier labels.
3. **Holdings overrides** - Adjust `data/3_processed_data/final_classification.json` after a run (for example, tweak `override_sector`), then rebuild. Changes persist via `data/2_fetched_data/yf_classification_cache.json`.
4. **Template updates** - Modify the PPT in `data/1_input_data/3_ppt_template/`, ensure placeholders retain their names, then run the audit command to confirm they are still detected.

### 5. Operational Tips
- Close any generated PPT before rebuilding; Windows locks the file while it is open.
- Keep dependencies isolated in a virtual environment (`python -m venv .venv` then `.\.venv\Scripts\Activate.ps1`).
- Ensure the Aggregated Amounts export includes the expected "Aggregated Amounts" sheet and column names (Date, Amount Client Currency, etc.).
- When offline or skipping AI usage, set `commentary.enabled` to `false`.
- `data/3_processed_data/commentary_generation_cache.json` stores the latest commentary output, while `data/2_fetched_data/news_cache.json` retains the newest headlines.
