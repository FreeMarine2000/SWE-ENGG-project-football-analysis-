"""
data_manager.py
───────────────
Kaggle dataset loader using developer credentials.
Users never need to log in or set anything up.

How it works:
  - Developer's Kaggle credentials are hardcoded here
  - First run: downloads dataset and caches it locally
  - Every run after: loads instantly from local cache
  - Users just run the script — nothing else needed

⚠ IMPORTANT FOR DEVELOPER:
   Replace KAGGLE_USERNAME and KAGGLE_KEY below with your own.
   Get your key from: https://www.kaggle.com/settings → API → Create New Token
"""

import os
import json
import pickle
import zipfile
import hashlib
import subprocess
import sys
import requests
import numpy as np
import pandas as pd
from pathlib import Path

# ════════════════════════════════════════════════════════════════
# ⚙ DEVELOPER CREDENTIALS — replace with your own
# ════════════════════════════════════════════════════════════════
KAGGLE_USERNAME = "rehan132334"
KAGGLE_KEY      = "KGAT_b824aabc25afc3e45978d5b866887ab2"

# Kaggle dataset to use
KAGGLE_DATASET  = "aleespinosa/soccer-match-event-dataset"

# Local cache directory — hidden folder in user's home
CACHE_DIR       = Path.home() / '.football_predictor' / 'data'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

WYSCOUT_DIR     = CACHE_DIR / 'wyscout'
WYSCOUT_READY   = WYSCOUT_DIR / '.download_complete'   # marker file


# ════════════════════════════════════════════════════════════════
# SECTION 1 — Credential injection (invisible to user)
# ════════════════════════════════════════════════════════════════

def _inject_credentials():
    """
    Inject developer credentials into the environment and
    kaggle.json — completely invisible to the user.
    Credentials come from this file, not from the user's machine.
    """
    # Set environment variables (kaggle library reads these)
    os.environ['KAGGLE_USERNAME'] = KAGGLE_USERNAME
    os.environ['KAGGLE_KEY']      = KAGGLE_KEY

    # Also write kaggle.json in case library needs file-based auth
    # Write to a temp location so we don't overwrite user's own credentials
    kaggle_dir  = Path.home() / '.kaggle'
    kaggle_json = kaggle_dir / 'kaggle.json'

    # Only write if user doesn't already have their own credentials
    if not kaggle_json.exists():
        kaggle_dir.mkdir(exist_ok=True)
        with open(kaggle_json, 'w') as f:
            json.dump({
                "username": KAGGLE_USERNAME,
                "key"     : KAGGLE_KEY
            }, f)
        os.chmod(kaggle_json, 0o600)


def _ensure_kaggle_installed():
    """Auto-install kaggle library if missing."""
    try:
        import kaggle
    except ImportError:
        print("  Installing kaggle library...")
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', 'kaggle', '-q']
        )
        print("  ✓ Installed")


# ════════════════════════════════════════════════════════════════
# SECTION 2 — Dataset download (runs once, cached forever)
# ════════════════════════════════════════════════════════════════

def _download_dataset():
    """
    Download Wyscout dataset from Kaggle using developer credentials.
    Only runs once — subsequent runs load from cache instantly.
    """
    if WYSCOUT_READY.exists():
        return True   # already downloaded

    print("\n" + "═" * 55)
    print("  First-time setup: downloading football dataset")
    print("  This runs ONCE and takes ~2 minutes")
    print("  After this, everything loads instantly from cache")
    print("═" * 55)

    _inject_credentials()
    _ensure_kaggle_installed()

    # Import after install
    import kaggle
    kaggle.api.authenticate()

    WYSCOUT_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = WYSCOUT_DIR / 'dataset.zip'

    print(f"\n  Downloading from Kaggle...")
    kaggle.api.dataset_download_files(
        KAGGLE_DATASET,
        path=str(WYSCOUT_DIR),
        unzip=False,        # we unzip manually for progress feedback
        quiet=False
    )

    # Find the downloaded zip
    zips = list(WYSCOUT_DIR.glob('*.zip'))
    if not zips:
        print("  ✗ Download failed — zip file not found")
        return False

    print(f"\n  Extracting files...")
    with zipfile.ZipFile(zips[0], 'r') as zf:
        members = zf.namelist()
        needed  = [m for m in members if 'England' in m or
                   m in ['players.json', 'teams.json']]
        for member in needed:
            zf.extract(member, WYSCOUT_DIR)
            print(f"  ✓ Extracted: {member}")

    # Remove zip to save space
    zips[0].unlink()

    # Write marker so we never re-download
    WYSCOUT_READY.write_text("download_complete")

    print("\n  ✓ Dataset ready — this message won't appear again")
    print("═" * 55 + "\n")
    return True


# ════════════════════════════════════════════════════════════════
# SECTION 3 — Data loading with smart caching
# ════════════════════════════════════════════════════════════════

def _load_raw_events():
    """
    Load events_England.csv (dataset now comes as CSV instead of JSON).
    Much faster than JSON parsing.
    """
    print("  Loading event data from CSV...")

    # Find the events CSV file
    csv_candidates = list(WYSCOUT_DIR.rglob('events_England.csv'))
    if not csv_candidates:
        print("  ✗ events_England.csv not found in downloaded data")
        return None
    
    events_path = csv_candidates[0]
    df = pd.read_csv(events_path, low_memory=False)
    
    print(f"  ✓ Loaded {len(df):,} events from CSV")
    return df


# ════════════════════════════════════════════════════════════════
# SECTION 4 — Public interface
# ════════════════════════════════════════════════════════════════

def get_events_df():
    """
    Main entry point — returns full events DataFrame.
    Handles download + caching automatically.
    Users just call this, nothing else needed.
    """
    # Step 1: ensure data is downloaded
    if not _download_dataset():
        print("✗ Dataset download failed.")
        print("  Check your internet connection and try again.")
        return None

    # Step 2: load and return
    df = _load_raw_events()
    if df is not None:
        print(f"  ✓ Loaded {len(df):,} events")
    return df


def get_player_events(player_name):
    """
    Returns all events for a specific player.
    Automatically handles download + caching.

    player_name — partial match supported (case insensitive)
                  e.g. 'Salah', 'salah', 'Mohamed Salah' all work
    """
    df = get_events_df()
    if df is None:
        return None

    player_df = df[
        df['player_name'].str.contains(player_name, case=False, na=False)
    ]

    if player_df.empty:
        print(f"\n⚠ Player '{player_name}' not found.")
        print("  Run list_available_players() to see all available names")
        return None

    # Show which exact name was matched
    matched_names = player_df['player_name'].unique()
    print(f"\n  Matched player(s): {list(matched_names)}")
    print(f"  Total events     : {len(player_df):,}")
    print(f"  Matches found    : {player_df['match_id'].nunique()}")

    return player_df


def list_available_players(search=None):
    """
    List all players in the dataset.
    search — optional filter string (e.g. 'Fer' finds Fernandes)
    """
    df = get_events_df()
    if df is None:
        return []

    all_players = sorted(df['player_name'].dropna().unique())

    if search:
        all_players = [p for p in all_players
                       if search.lower() in p.lower()]
        print(f"\nPlayers matching '{search}':")
    else:
        print(f"\nAll available players ({len(all_players)} total):")

    for i, name in enumerate(all_players, 1):
        print(f"  {i:4d}. {name}")

    return all_players


def list_available_teams():
    """List all teams in the dataset."""
    df = get_events_df()
    if df is None:
        return []

    teams = sorted(df['team_name'].dropna().unique())
    print(f"\nTeams in dataset ({len(teams)} total):")
    for t in teams:
        print(f"  - {t}")
    return teams


def get_cache_info():
    """Show what's cached and how much space it's using."""
    print("\n📦 Cache info:")
    print(f"  Location : {CACHE_DIR}")

    total_size = sum(
        f.stat().st_size for f in CACHE_DIR.rglob('*') if f.is_file()
    )
    print(f"  Size     : {total_size / 1024 / 1024:.1f} MB")
    print(f"  Status   : {'✓ Ready' if WYSCOUT_READY.exists() else '⚠ Not downloaded'}")


def clear_cache():
    """Clear cache to force fresh download on next run."""
    import shutil
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        print("✓ Cache cleared — data will re-download on next run")


# ════════════════════════════════════════════════════════════════
# SECTION 5 — Credential validation on import
# ════════════════════════════════════════════════════════════════

def _validate_credentials():
    """Warn developer if they forgot to set credentials."""
    if KAGGLE_USERNAME == "your_kaggle_username" or \
       KAGGLE_KEY      == "your_kaggle_api_key":
        print("\n" + "!" * 55)
        print("  ⚠ WARNING: Kaggle credentials not set in data_manager.py")
        print("  Open data_manager.py and replace:")
        print("    KAGGLE_USERNAME = 'your_kaggle_username'")
        print("    KAGGLE_KEY      = 'your_kaggle_api_key'")
        print("  Get your key: https://www.kaggle.com/settings → API")
        print("!" * 55 + "\n")
        return False
    return True

print(_validate_credentials())