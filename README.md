# ReportGenerator

Automates creation of a PowerPoint investment factsheet by merging portfolio inputs, Yahoo Finance market data, custom charts, and LLM-powered commentary.

## TL;DR - Quick Start
- Install Python 3.11+ and activate a virtual environment.
- `pip install -r requirements.txt`
- **Export and upload the latest Aggregated Amounts file** covering the period from inception date to the last day of the current month into `data/1_input_data/1_raw_financials/`. The newest `AggregatedAmounts_*.xlsx` is loaded automatically.
- Keep mapping CSVs under `data/1_input_data/2_mappings/` and the PPT template under `data/1_input_data/3_ppt_template/`.
- **Update `config.yaml`** with the new factsheet date (`as_of`) to match the last day of the month (e.g., `2025-10-31` for October 2025).
- Set `OPENAI_API_KEY` in your environment (`$env:OPENAI_API_KEY="sk-..."` on PowerShell) or add it to a `.env` file in the project root (auto-loaded via `python-dotenv`).
- Build the deck: `python -m factsheet.app build --config config.yaml`
- The builder combines raw inputs with fetched Yahoo Finance data cached in `data/2_fetched_data/`, writes processed artefacts to `data/3_processed_data/`, and saves the finished PPT and charts in `output/`.
- Close any generated PPT before rebuilding to avoid Windows file locks.

## Deep Dive

### 1. Pipeline Overview
1. **Configuration** - `config.yaml` defines report dates, file locations, output template, lookup tables, and commentary settings.
2. **Input prep (`data/1_input_data`)** - Raw financials (Aggregated Amounts from inception to end of current month), mapping tables, and the PPT template are loaded from their respective subfolders.
3. **Data ingest (`factsheet.data`)** - Reads the latest Aggregated Amounts workbook using only "Position Value" rows for holdings distribution, fetches Yahoo Finance classifications and pricing (stored in `data/2_fetched_data/`), and merges everything into a unified dataset.
4. **Portfolio metrics (`factsheet.calc`)** - Calculates performance metrics, allocation splits, and top holdings based on the combined dataset.
5. **Monthly performance analysis (`factsheet.data`)** - Calculates monthly returns for each instrument using "Percent return per Instrument" rows, then identifies top 4 winners and losers based on value contribution (weight × return) rather than just percentage return.
6. **Chart assembly (`factsheet.charts`)** - Builds line, bar, and allocation charts, exporting PNG fallbacks if needed.
7. **Commentary generation (`factsheet.commentary`)** - Intelligent news aggregation and filtering:
   - Fetches general market news from Google News RSS feeds
   - Applies AI filtering to select the most relevant market context articles (default: top 10)
   - Fetches stock-specific news from Google News for the top 4 winners and losers by value impact
   - Applies AI filtering per stock to select news that explains why each stock gained/lost value (default: top 5 per stock)
   - Combines filtered market and stock news (no additional filtering)
   - Generates commentary using GPT-4o-mini with context about winners/losers and their value contributions
   - All prompts/responses cached in `data/3_processed_data/commentary_generation_cache.json`
8. **PPT templating (`factsheet.ppt`)** - Injects tables, charts, and commentary into the PPT template while preserving formatting.
9. **Outputs** - The finished deck lands in `output/`, charts are saved to `dist/charts/`, and supporting artefacts (classification review JSON, commentary history, news cache) land in `data/2_fetched_data/` and `data/3_processed_data/`.

Use the audit helper whenever you tweak the template:  
`python -m factsheet.app audit --config config.yaml`

### 2. Data Requirements

**IMPORTANT: Aggregated Amounts File**

The Excel file must contain data from **inception date through the last day of the current month**. For example:
- If generating a factsheet for October 2025, the file should span from inception (e.g., 2021-09-06) through 2025-10-31
- The file must include these Amount Type Names in the "Aggregated Amounts" sheet:
  - **"Position Values"** - Used for calculating holdings weights and portfolio distribution
  - **"Percent return per Instrument"** - Used for calculating monthly performance and identifying winners/losers

**Monthly Update Process:**
1. Export new Aggregated Amounts file covering inception to end of current month
2. Place file in `data/1_input_data/1_raw_financials/` (system automatically picks newest `AggregatedAmounts_*.xlsx`)
3. Update `config.yaml` field `as_of` to the last day of the month (e.g., `2025-10-31`)
4. Run the build

### 3. Required Inputs and Key Artefacts
| File / Folder | Purpose |
| --- | --- |
| `data/1_input_data/1_raw_financials/` | Drop the latest Aggregated Amounts export here (inception → end of month); the build picks the newest `AggregatedAmounts_*.xlsx`. |
| `data/1_input_data/2_mappings/instrument_overrides.csv`, `region_map.csv`, `sector_aliases.csv` | Map tickers/sectors to the display buckets used in charts and tables. |
| `data/1_input_data/3_ppt_template/Kukula Fact Sheet Growth 20250830.pptx` | PowerPoint template providing layout and placeholder names. |
| `data/2_fetched_data/yf_classification_cache.json` | Persistent cache for holdings classifications fetched from Yahoo Finance. |
| `data/2_fetched_data/news_cache.json` | Latest filtered news articles (market + stock-specific) used for commentary. |
| `data/3_processed_data/final_classification.json` | Generated after each run to review/edit resolved holdings; feeds back into the cache. |
| `data/3_processed_data/commentary_generation_cache.json` | Stores the latest commentary context (winners/losers, monthly returns, news), prompt, and output. |

### 4. Key Sections in `config.yaml`
```yaml
as_of: 2025-10-31
start_date: 2021-09-06
base_currency: USD

benchmarks:
  - name: ACWI
    ticker: ACWI
  - name: S&P 500
    ticker: ^GSPC

fx:
  source: ecb

output:
  ppt_template: ./data/1_input_data/3_ppt_template/Kukula Fact Sheet Growth 20250830.pptx
  ppt_out_dir: ./output
  charts_dir: ./dist/charts

data_sources:
  aggregated_amounts_dir: ./data/1_input_data/1_raw_financials
  aggregated_amounts_pattern: AggregatedAmounts_*.xlsx

tables:
  top_holdings_limit: 10
  instrument_overrides_file: ./data/1_input_data/2_mappings/instrument_overrides.csv
  region_map_file: ./data/1_input_data/2_mappings/region_map.csv
  classification_cache: ./data/2_fetched_data/yf_classification_cache.json
  sector_aliases_file: ./data/1_input_data/2_mappings/sector_aliases.csv
  classification_review_file: ./data/3_processed_data/final_classification.json
```

- `base_currency` sets the reporting currency (default: USD).
- `benchmarks` defines the benchmark indices used for performance comparison.
- `fx.source` specifies the FX rate provider (ecb = European Central Bank).
- `tables.top_holdings_limit` caps the holdings table.
- The finished PPT is saved to `output.ppt_out_dir` (default: `./output`).
- Charts are exported to `output.charts_dir` (default: `./dist/charts`).
- `tables.classification_review_file` is regenerated each cycle for manual review and feeds back into the persisted cache at `tables.classification_cache`.

#### Commentary Configuration
```yaml
commentary:
  enabled: true
  model: gpt-4o-mini              # OpenAI model for AI filtering and commentary generation
  max_tokens: 300                 # Increased to ensure complete sentences
  temperature: 0.6
  mode: mixed                     # Commentary mode (mixed = market + holdings news)
  max_per_feed: 50                # Fetch up to 50 articles per RSS feed
  news_per_ticker: 5              # Select top 5 news per stock after AI filtering
  max_headlines: 20               # Final limit for commentary (after AI + keyword filtering)
  max_macro_headlines: 10         # AI filters general market news to top 10
  max_holdings_headlines: 10      # Maximum headlines for holdings-specific news
  max_news_age_days: 30           # Only fetch news from last 30 days
  rss_feeds:                      # RSS feeds for market context
    # Google News
    - https://news.google.com/rss/search?q=stock+market&hl=en-US&gl=US&ceid=US:en
    - https://news.google.com/rss/search?q=stocks&hl=en-US&gl=US&ceid=US:en
    - https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx6TVdZU0FtVnVHZ0pWVXlnQVAB?hl=en-US&gl=US&ceid=US:en
    # Yahoo Finance
    - https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC&region=US&lang=en-US
    - https://feeds.finance.yahoo.com/rss/2.0/headline?s=^DJI&region=US&lang=en-US
    - https://feeds.finance.yahoo.com/rss/2.0/headline?s=^IXIC&region=US&lang=en-US
    # Financial News Sites
    - https://www.cnbc.com/id/100003114/device/rss/rss.html
    - https://www.cnbc.com/id/10000664/device/rss/rss.html
    - https://www.cnbc.com/id/10001147/device/rss/rss.html
    - https://feeds.marketwatch.com/marketwatch/topstories/
    - https://feeds.marketwatch.com/marketwatch/marketpulse/
    - https://www.investing.com/rss/news.rss
    - https://www.investing.com/rss/news_25.rss
    - https://seekingalpha.com/feed.xml
    # Reuters & Bloomberg-style
    - https://feeds.a.dj.com/rss/RSSMarketsMain.xml
    - https://feeds.a.dj.com/rss/WSJcomUSBusiness.xml
  exclude_keywords:
    - boeing                      # Exclude specific tickers/topics
  history_file: data/3_processed_data/commentary_generation_cache.json
  news_cache: data/2_fetched_data/news_cache.json
```

**How Commentary Works:**
1. **Winner/Loser Selection** - System identifies top 4 winners and losers by **value contribution** (weight × monthly return), not just percentage return
2. **Market News** - Fetches general market news from multiple RSS feeds (Google News, Yahoo Finance, CNBC, MarketWatch, etc.), then AI filters to select top 10 most relevant articles for portfolio context
3. **Stock News** - Fetches Google News for each winner/loser, then AI filters per stock to select top 5 articles explaining why it gained/lost value
4. **Commentary Generation** - GPT-4o-mini receives:
   - Portfolio performance (MTD vs benchmark)
   - Winners/losers with their returns, weights, and value contributions
   - Filtered market news + filtered stock-specific news
   - Instructions to comment on underperformance only if portfolio is behind benchmark

**Configuration Options:**
- Set `enabled: false` to craft commentary manually
- Adjust `max_macro_headlines` to control market news volume (default: 10)
- Adjust `news_per_ticker` to control per-stock news volume (default: 5)
- Adjust `max_headlines` to set final limit for all news (default: 20)
- Set `mode: mixed` for market + holdings news (default), or other modes as supported
- Add/remove RSS feeds in `rss_feeds` to customize news sources
- Use `exclude_keywords` to filter out specific tickers or topics

### 5. Automatic Ticker Resolution

The system **automatically** converts exchange-specific tickers to Yahoo Finance format without manual overrides. No matter which exchange a stock trades on, it will work automatically.

#### How It Works

Your raw financial data includes tickers with MIC codes (Market Identifier Codes) like:
- `PIK:xjse` (Johannesburg)
- `0700:xhkg` (Hong Kong)
- `AAPL:xnas` (NASDAQ)
- `VOW:xetr` (Frankfurt)
- `UMI:xbru` (Brussels)

The system automatically maps these to Yahoo Finance format:
- `PIK:xjse` → `PIK.JO`
- `0700:xhkg` → `0700.HK` (with leading zero normalization)
- `AAPL:xnas` → `AAPL` (US stocks need no suffix)
- `VOW:xetr` → `VOW.DE`
- `UMI:xbru` → `UMI.BR`

#### Supported Exchanges

The system has built-in support for **60+ global exchanges** including:
- **North America**: NASDAQ (xnas), NYSE (xnys), TSX (xtse)
- **Europe**: London (xlon), Paris (xpar), Frankfurt (xetr), Amsterdam (xams), Brussels (xbru), Copenhagen (xcse), Stockholm (xsto), Zurich (xswx), Milan (xmil), Madrid (xmad), and more
- **Asia Pacific**: Hong Kong (xhkg), Shanghai (xshg), Shenzhen (xshe), Tokyo (xtks), Korea (xkrx), Taiwan (xtai), Singapore (xses), Australia (xasx), India (xnse/xbom)
- **Middle East & Africa**: Johannesburg (xjse), Tel Aviv (xtae), Dubai (xdfm), Saudi Arabia (xsau)
- **Latin America**: São Paulo (bvmf), Mexico (xmex), Buenos Aires (xbue)

**No manual configuration needed** - buy any stock on any supported exchange and the system handles it automatically!

#### When to Use instrument_overrides.csv

The `instrument_overrides.csv` file is now **optional** and only needed for exceptional cases:
- Stocks where the ticker symbol doesn't match the standard conversion (e.g., preference shares with special suffixes)
- Manual sector/region classification overrides
- Legacy holdings without exchange codes

For 99% of use cases, the file can remain empty - the system handles everything automatically.

### 6. Manual Overrides and Customisation
1. **Sector and region mapping** - Edit the CSVs in `data/1_input_data/2_mappings/` to change how holdings roll up into display buckets.
2. **Fallback sector names** - Update `sector_aliases.csv` for any instruments that need friendlier labels.
3. **Holdings overrides** - Adjust `data/3_processed_data/final_classification.json` after a run (for example, tweak `override_sector`), then rebuild. Changes persist via `data/2_fetched_data/yf_classification_cache.json`.
4. **Template updates** - Modify the PPT in `data/1_input_data/3_ppt_template/`, ensure placeholders retain their names, then run the audit command to confirm they are still detected.

### 7. Operational Tips
- **Data Update:** Always export Aggregated Amounts from inception through the last day of the current month, then update `as_of` in `config.yaml` to match the month-end date.
- **File Requirements:** The Aggregated Amounts sheet must include both "Position Values" (for holdings) and "Percent return per Instrument" (for monthly performance analysis).
- Close any generated PPT before rebuilding; Windows locks the file while it is open.
- Keep dependencies isolated in a virtual environment (`python -m venv .venv` then `.\.venv\Scripts\Activate.ps1`).
- When offline or skipping AI usage, set `commentary.enabled` to `false`.
- **Cache files:**
  - `data/2_fetched_data/yf_classification_cache.json` persists Yahoo Finance classifications to avoid repeated API calls
  - `data/2_fetched_data/news_cache.json` stores the filtered news articles (market + stock-specific)
  - `data/3_processed_data/commentary_generation_cache.json` stores the latest commentary context (winners/losers, returns, news) and output
  - `data/3_processed_data/final_classification.json` stores the final classification review for manual edits
