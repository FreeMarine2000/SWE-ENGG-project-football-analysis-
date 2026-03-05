from data_utils import (
    check_connection, select_team, get_completed_matches, get_block_map,
    get_players, collect_player_rows, build_features, build_prediction_input,
    accuracy_report, FEATURE_COLS, TARGET_COLS, understat
)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

# ── Setup ───────────────────────────────────────────────────────
check_connection()
team_name_url, team_name_data, league_team_data = select_team()

# ── Completed matches ───────────────────────────────────────────
completed_matches = get_completed_matches(team_name_url, '2025')
print(f"\nTotal completed matches in 2025 : {len(completed_matches)}")

mu_history = None
for tid, tinfo in league_team_data.items():
    if tinfo.get('title', '') == team_name_data:
        mu_history = tinfo.get('history', [])
        break
block_map = get_block_map(completed_matches, mu_history)

# ── Last 20 completed matches ───────────────────────────────────
LAST_N         = 20
recent_matches = completed_matches[-LAST_N:]
print(f"Using last {len(recent_matches)} completed matches for training")
print(f"Rolling form window : last 5 matches  (HIGH priority)")
print(f"Block history       : all {len(recent_matches)} matches (LOW priority)")

# ── Player selection ────────────────────────────────────────────
player_name = get_players(team_name_url, team_name_data, recent_matches)

# ── Collect data ────────────────────────────────────────────────
rows         = collect_player_rows(player_name, recent_matches, block_map)
player_stats = pd.DataFrame(rows)
print(f"\nTotal rows collected: {len(player_stats)}")
if player_stats.empty:
    print(f"No data found for: {player_name}"); exit(1)

# ── Build features ──────────────────────────────────────────────
player_stats       = build_features(player_stats)
position_map       = dict(zip(player_stats['position'], player_stats['position_encoded']))
position_map_lower = {k.lower(): v for k, v in position_map.items()}
print(f"\nPosition map: {position_map}")
print("\nFeature priority breakdown:")
print("  [HIGH] rolling_xG_5, rolling_goals_5, rolling_assists_5,")
print("         rolling_xA_5, rolling_key_passes_5, rolling_shots_5, rolling_time_5")
print("  [LOW]  hist_xG_vs_block, hist_goals_vs_block, hist_xGChain_vs_block")
print("  [CTX]  block_encoded, home_encoded, opp_ppda, position_encoded")

X = player_stats[FEATURE_COLS].astype(float).values
y = player_stats[TARGET_COLS].astype(float).values
if len(X) < 4:
    print("Not enough data."); exit(1)

X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTrain: {len(X_tr)} | Val: {len(X_va)}")

# ── Random Forest ────────────────────────────────────────────────
print("\n🌲 Training Random Forest...")
model = MultiOutputRegressor(
    RandomForestRegressor(
        n_estimators=200, max_depth=5,
        min_samples_leaf=2, random_state=42, n_jobs=-1
    )
)
model.fit(X_tr, y_tr)
y_pred = np.clip(model.predict(X_va), 0, None)

accuracy_report(y_va, y_pred, "RANDOM FOREST")

# ── Feature importance ──────────────────────────────────────────
print("\nFeature Importance (avg across all targets):")
importances = np.mean([est.feature_importances_ for est in model.estimators_], axis=0)
for feat, imp in sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1]):
    tag = "[HIGH]" if 'rolling' in feat else "[LOW] " if 'hist' in feat else "[CTX] "
    print(f"  {tag} {feat:<28} {'█' * int(imp * 60)} {imp:.4f}")

# ── Prediction ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PREDICTION")
print("=" * 60)
opponent_block = input("Opponent block type (mid_block: 0 / low_block: 1): ").strip()

lr_model = LinearRegression()
lr_model.fit(player_stats['block_encoded'].values.reshape(-1, 1), player_stats['opp_ppda'].values)
predicted_ppda = lr_model.predict([[int(opponent_block)]])[0]
print(f"Predicted PPDA: {predicted_ppda:.2f}")

home_away      = input("Home or Away? (1=Home / 0=Away): ").strip()
print(f"Available positions: {list(position_map.keys())}")
position_input = input("Enter player position: ").strip()
position_code  = position_map_lower.get(position_input.lower(), 0)
if position_input.lower() not in position_map_lower:
    print("  ⚠ Unknown position, defaulting to 0")

input_row   = build_prediction_input(
    player_stats, opponent_block, home_away, predicted_ppda, position_code
)
predictions = np.clip(model.predict(input_row), 0, None)

print(f"\nPredicted stats for {player_name}:")
print(f"  (Form: last 5 matches  |  Block history: last {LAST_N} matches)")
print("-" * 40)
for i, col in enumerate(TARGET_COLS):
    print(f"  {col:<14}: {predictions[0][i]:.2f}")