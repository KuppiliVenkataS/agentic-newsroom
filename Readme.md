# Agentic Newsroom — Ingestion Module

## One-time setup

```bash
# 1. Clone or copy this folder to your Mac Mini
cd ~/projects
# (copy agentic_newsroom here)

# 2. Create virtual environment
python3.12 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Mount your external disk and create the storage folder
mkdir -p /Volumes/OilNewsDB/agentic_newsroom

# 5. Set your EIA API key in config/settings.py
#    Get a free key at: https://www.eia.gov/opendata/
#    Replace: EIA_API_KEY = "add key here"

# 6. If your external disk mounts at a different path, update STORAGE_ROOT in config/settings.py
```

## Run manually

```bash
source venv/bin/activate
python run_ingestion.py
```

## Set up cron (runs every 12 hours at 06:00 and 18:00)

```bash
crontab -e
```

Add this line (adjust paths to match where you placed the project):

```
0 6,18 * * * /Users/YOUR_USERNAME/projects/agentic_newsroom/venv/bin/python /Users/YOUR_USERNAME/projects/agentic_newsroom/run_ingestion.py >> /Volumes/OilNewsDB/agentic_newsroom/logs/cron.log 2>&1
```

Save and exit. Verify it's registered:
```bash
crontab -l
```

## Output on disk

```
/Volumes/OilNewsDB/agentic_newsroom/
  raw/
    2026/
      05/
        2026-05-21_06-00-00_run.json   ← one file per run, keep forever
  logs/
    audit_2026-05.jsonl                ← one line per run, quick status check
    ingestion.log                      ← rolling verbose log
    cron.log                           ← stdout from cron runs
  dedup.db                             ← SQLite dedup registry
```

## Check recent runs

```bash
tail -5 /Volumes/OilNewsDB/agentic_newsroom/logs/audit_$(date +%Y-%m).jsonl | python3 -m json.tool
```

## What each file does

| File | Purpose |
|---|---|
| `config/settings.py` | All config: paths, API keys, feed URLs |
| `ingestion/dedup.py` | SQLite registry — prevents duplicate articles |
| `ingestion/rss_fetcher.py` | Fetches and parses all RSS feeds |
| `ingestion/eia_fetcher.py` | Pulls oil price and inventory data from EIA API |
| `ingestion/archive_writer.py` | Saves each run as a JSON file on your external disk |
| `ingestion/audit_logger.py` | Writes one audit line per run for quick history checks |
| `run_ingestion.py` | Entry point — orchestrates everything |

## Adding or removing RSS feeds

Edit the `RSS_FEEDS` dict in `config/settings.py`. No other file needs to change.

## Notes

- The script is safe to run multiple times. Dedup prevents duplicate articles.
- If your external disk is unmounted, the run will fail with a clear error in the audit log.
- EIA data is skipped gracefully if the API key is not set.
- Exit code 1 on full failure, so cron can be wired to an alert if needed.