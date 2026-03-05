from understatapi import UnderstatClient
import pandas as pd
import requests
import numpy as np

understat = UnderstatClient()

# ── Connection check ────────────────────────────────────────────
def check_connection():
    try:
        test = requests.get("https://understat.com", timeout=100)
        print(f"✓ understat.com is reachable (status: {test.status_code})")
    except requests.exceptions.ConnectTimeout:
        print("✗ Connection timed out."); exit(1)
    except requests.exceptions.ConnectionError as e:
        print(f"✗ Cannot connect: {e}"); exit(1)

# ── Helpers ─────────────────────────────────────────────────────
def classify_block(ppda_value):
    return "mid_block" if ppda_value <= 12 else "low_block"

def get_completed_matches(team_url, season):
    all_matches = understat.team(team=team_url).get_match_data(season=season)
    return [m for m in all_matches if m.get('isResult', False) == True]

def get_block_map(completed, history):
    block_map = {}
    if not history:
        return block_map
    date_to_id = {}
    for m in completed:
        dt = m.get('datetime', '')
        date_to_id[dt.split(' ')[0] if ' ' in dt else dt] = m['id']
    for h in history:
        d   = h.get('date', '').split(' ')[0]
        mid = date_to_id.get(d)
        if not mid:
            continue
        ppda     = h.get('ppda', {})
        att      = float(ppda.get('att', 0))
        def_     = float(ppda.get('def', 1))
        ppda_val = att / def_ if def_ > 0 else 10.0
        block_map[mid] = {
            'ppda'      : round(ppda_val, 2),
            'block_type': classify_block(ppda_val)
        }
    return block_map

# ── Team selection ──────────────────────────────────────────────
def select_team():
    print("ENTER THE LEGUE NAME(EPL, LaLiga, SerieA, Bundesliga, Ligue1):")
    legue_name=input().strip()
    if legue_name not in ['EPL', 'LaLiga', 'SerieA', 'Bundesliga', 'Ligue1']:
        print("Invalid legue name. Please enter one of: EPL, LaLiga, SerieA, Bundesliga, Ligue1")
        exit(1)
    league_team_data = understat.league(league=legue_name).get_team_data(season='2025')
    teams_available  = [(tid, t.get('title', '')) for tid, t in league_team_data.items()]
    teams_available.sort(key=lambda x: x[1])  # sort by team name
    print(f"\nAvailable teams in {legue_name} 2025:")
    for idx, (team_id, team_name) in enumerate(teams_available, 1):
        print(f"  {idx}. {team_name} (ID: {team_id})")
    team_input     = input("\nEnter team name exactly as shown above: ").strip()
    team_name_url  = team_input.replace(" ", "_")
    team_name_data = team_input
    return team_name_url, team_name_data, league_team_data

# ── Show available players ──────────────────────────────────────
def get_players(team_name_url, team_name_data, recent_matches):
    print(f"\n--- Players in {team_name_data} (last 5 matches) ---")
    # Dictionary to store unique players: {player_name: player_id}
    player_dict = {}
    for i in range(max(0, len(recent_matches) - 5), len(recent_matches)):
        try:
            mid        = recent_matches[i]["id"]
            match_info = recent_matches[i]
            roster     = understat.match(match=mid).get_roster_data()
            rm_side    = match_info.get('side', '')
            if rm_side not in ['h', 'a']:
                title   = match_info.get('title', '')
                rm_side = 'h' if team_name_data in title.split(' - ')[0] else 'a'
            print(f"  {match_info.get('title', 'N/A')} | side: {rm_side} | match_id: {mid}")
            for pid, info in roster[rm_side].items():
                player_name = info['player']
                # Keep only the first occurrence of each player
                if player_name not in player_dict:
                    player_dict[player_name] = pid
        except Exception as e:
            print(f"  Match error: {e}")
    
    # Sort players alphabetically
    sorted_players = sorted(player_dict.items(), key=lambda x: x[0])
    for idx, (name, player_id) in enumerate(sorted_players, 1):
        print(f"  {idx}. {name} (ID: {player_id})")
    print(f"\nTotal players found: {len(sorted_players)}")
    return input("\nEnter player name: ").strip()

# ── Collect raw rows ────────────────────────────────────────────
def collect_player_rows(player_name, recent_matches, block_map):
    rows = []
    for match_idx in range(len(recent_matches)):
        match_id   = recent_matches[match_idx]["id"]
        match_info = recent_matches[match_idx]
        title      = match_info.get('title', '')
        home_team  = title.split(' - ')[0].strip() if ' - ' in title else title
        away_team  = title.split(' - ')[1].strip() if ' - ' in title else ''
        try:
            roster_data = understat.match(match=match_id).get_roster_data()
        except Exception as e:
            print(f"  Skipping match {match_id}: {e}"); continue
        for side, label in [('h', 'Home'), ('a', 'Away')]:
            opponent = away_team if side == 'h' else home_team
            for pid, info in roster_data[side].items():
                if info['player'] == player_name:
                    block_info = block_map.get(match_id, {'ppda': 10.0, 'block_type': 'mid_block'})
                    rows.append({
                        'goals'         : float(info['goals']),
                        'assists'       : float(info['assists']),
                        'xG'            : float(info['xG']),
                        'xA'            : float(info['xA']),
                        'shots'         : float(info['shots']),
                        'key_passes'    : float(info['key_passes']),
                        'xGChain'       : float(info['xGChain']),
                        'xGBuildup'     : float(info['xGBuildup']),
                        'time'          : float(info['time']),
                        'position'      : info['position'],
                        'Home or Away'  : label,
                        'opponent'      : opponent,
                        'match_id'      : match_id,
                        'opp_ppda'      : block_info['ppda'],
                        'opp_block_type': block_info['block_type']
                    })
                    print(f"  ✓ {title} | pos: {info['position']} | "
                          f"xG: {info['xG']} | goals: {info['goals']} | "
                          f"block: {block_info['block_type']}")
    return rows

# ── Feature engineering ─────────────────────────────────────────
def build_features(player_stats):
    """
    HIGH PRIORITY  — Rolling form last 5 matches (7 features)
    LOW PRIORITY   — Historical performance vs same block type (3 features)
    CONTEXT        — Match context (4 features)
    TOTAL          — 14 features
    """
    df = player_stats.copy()

    # ── Encodings ──────────────────────────────────────────────
    df['block_encoded']    = df['opp_block_type'].map({'mid_block': 0, 'low_block': 1})
    df['home_encoded']     = df['Home or Away'].map({'Home': 1, 'Away': 0})
    df['position_encoded'] = pd.Categorical(df['position']).codes

    # ── HIGH PRIORITY: Rolling form — window=5 ─────────────────
    for col in ['xG', 'goals', 'assists', 'xA', 'key_passes', 'shots', 'time']:
        df[f'rolling_{col}_5'] = (
            df[col].astype(float)
            .rolling(window=5, min_periods=1)
            .mean()
        )

    # ── LOW PRIORITY: Historical performance vs same block ──────
    # For each row, look back at ALL rows with the same block type
    # and compute the player's average output in those matches
    for col in ['xG', 'goals', 'xGChain']:
        hist_col = f'hist_{col}_vs_block'
        df[hist_col] = 0.0
        for i in range(len(df)):
            current_block = df.iloc[i]['block_encoded']
            # Only look at rows BEFORE current index (no data leakage)
            past_rows = df.iloc[:i]
            same_block = past_rows[past_rows['block_encoded'] == current_block]
            if len(same_block) > 0:
                df.at[df.index[i], hist_col] = same_block[col].astype(float).mean()
            else:
                # No past data for this block — use overall mean as fallback
                df.at[df.index[i], hist_col] = df[col].astype(float).mean()

    return df

# ── Final feature and target columns ───────────────────────────
# 7 rolling (HIGH) + 3 block-history (LOW) + 4 context = 14 total
FEATURE_COLS = [
    # HIGH PRIORITY — rolling form last 5
    'rolling_xG_5', 'rolling_goals_5', 'rolling_assists_5',
    'rolling_xA_5', 'rolling_key_passes_5', 'rolling_shots_5', 'rolling_time_5',
    # LOW PRIORITY — historical vs same block
    'hist_xG_vs_block', 'hist_goals_vs_block', 'hist_xGChain_vs_block',
    # CONTEXT
    'block_encoded', 'home_encoded', 'opp_ppda', 'position_encoded'
]

TARGET_COLS = ['goals', 'assists', 'xG', 'xA', 'key_passes', 'xGChain', 'xGBuildup']

# ── Accuracy report ─────────────────────────────────────────────
def accuracy_report(y_true, y_pred, model_name):
    from sklearn.metrics import r2_score
    print("\n" + "=" * 60)
    print(f"{model_name} — ACCURACY REPORT")
    print("=" * 60)
    for i, col in enumerate(TARGET_COLS):
        mae  = np.mean(np.abs(y_pred[:, i] - y_true[:, i]))
        rmse = np.sqrt(np.mean((y_pred[:, i] - y_true[:, i]) ** 2))
        if np.std(y_true[:, i]) < 1e-8:
            r2_str = "N/A (no variance)"
        else:
            r2_str = f"{r2_score(y_true[:, i], y_pred[:, i]):.4f}"
        mean_true    = np.mean(y_true[:, i])
        accuracy_pct = max(0, min((1 - mae / (mean_true + 1e-8)) * 100, 100))
        print(f"\n  {col:<14} MAE: {mae:.4f} | RMSE: {rmse:.4f} | "
              f"R²: {r2_str} | Acc: {accuracy_pct:.1f}%")
    print(f"\n  Overall MAE : {np.mean(np.abs(y_pred - y_true)):.4f}")
    print("=" * 60)

# ── Prediction input builder ────────────────────────────────────
def build_prediction_input(player_stats, opponent_block, home_away,
                            predicted_ppda, position_code):
    """
    Build the prediction row using:
    - Last 5 match rolling averages (high priority form)
    - Historical block averages computed from all training data (low priority)
    - User-provided match context
    """
    df   = player_stats.copy()
    last = df.iloc[-1]   # most recent match row (already has all engineered features)

    # Block history for the selected block type
    same_block_rows = df[df['block_encoded'] == int(opponent_block)]
    hist_xG    = same_block_rows['xG'].astype(float).mean() if len(same_block_rows) > 0 else df['xG'].astype(float).mean()
    hist_goals = same_block_rows['goals'].astype(float).mean() if len(same_block_rows) > 0 else df['goals'].astype(float).mean()
    hist_xGChain = same_block_rows['xGChain'].astype(float).mean() if len(same_block_rows) > 0 else df['xGChain'].astype(float).mean()

    input_row = np.array([[
        # HIGH PRIORITY — rolling form (use last row's rolling values)
        last['rolling_xG_5'], last['rolling_goals_5'], last['rolling_assists_5'],
        last['rolling_xA_5'], last['rolling_key_passes_5'], last['rolling_shots_5'],
        last['rolling_time_5'],
        # LOW PRIORITY — block history
        hist_xG, hist_goals, hist_xGChain,
        # CONTEXT — user inputs for upcoming match
        int(opponent_block), int(home_away), predicted_ppda, position_code
    ]], dtype=float)

    return input_row