import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from urllib.request import urlopen, Request
from PIL import Image
from io import BytesIO
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

def clean_name(name):
    """Remove non-Latin characters (Arabic/Urdu etc.) and extra whitespace."""
    cleaned = re.sub(r'[^\x00-\x7F\xC0-\xFF]+', '', str(name))
    return re.sub(r'\s+', ' ', cleaned).strip()

# ── Load & Preprocess ────────────────────────────────────────────────────────
players_data = pd.read_csv(r"C:\Users\Rehan Ahmed\Downloads\new-players-data-full.csv")
players_data['version'] = pd.to_datetime(players_data['version'], dayfirst=True)
players_data['dob'] = pd.to_datetime(players_data['dob'], dayfirst=True)
players_data['age'] = players_data['version'].dt.year - players_data['dob'].dt.year

# Parse market value for use as prediction target
def parse_currency(series):
    s = series.str.replace('€', '', regex=False).str.strip()
    multiplier = s.str.extract(r'([MK])$', expand=False).map({'M': 1_000_000, 'K': 1_000}).fillna(1)
    numbers = s.str.replace(r'[MK]$', '', regex=True).astype(float)
    return (numbers * multiplier).astype(float)

players_data['value_numeric'] = parse_currency(players_data['value'])
players_data['log_value'] = np.log1p(players_data['value_numeric'])  # log-scale for better prediction
players_data['preferred_foot'] = players_data['preferred_foot'].fillna('Right')

# ── Encoding ──────────────────────────────────────────────────────────────────
le_position = LabelEncoder()
le_name = LabelEncoder()
le_foot = LabelEncoder()

players_data['position_encoded'] = le_position.fit_transform(players_data['best_position'])
players_data['name_encoded'] = le_name.fit_transform(players_data['name'])
players_data['foot_encoded'] = le_foot.fit_transform(players_data['preferred_foot'])

# ── Feature Engineering ───────────────────────────────────────────────────────
# Only use stable features that won't change over time
players_data['num_positions'] = players_data['positions'].str.split(',').str.len()

# ── Career Curve Bias (data-driven, no plateau) ─────────────────────────────
# Step 1: Derive peak age per position via weighted quadratic fit
#   mean_rating(age) ≈ a·age² + b·age + c  →  peak = -b / (2a)
peak_age_map = {}
for pos, grp in players_data.groupby('best_position'):
    age_rating = grp.groupby('age')['overall_rating'].agg(['mean', 'count'])
    age_rating = age_rating[age_rating['count'] >= 10]
    if len(age_rating) >= 3:
        coeffs = np.polyfit(age_rating.index, age_rating['mean'], 2,
                            w=np.sqrt(age_rating['count']))
        a_coeff, b_coeff, _ = coeffs
        if a_coeff < 0:
            peak_age_map[pos] = np.clip(-b_coeff / (2 * a_coeff), 29, 35)
        else:
            peak_age_map[pos] = 29.0
    else:
        peak_age_map[pos] = 29.0
print("Data-derived peak ages per position:")
for pos in sorted(peak_age_map):
    print(f"  {pos}: {peak_age_map[pos]:.1f}")

# Step 2: Piecewise career curve — GROW until peak, then DECLINE (no plateau)
#   Growth : age 16 → peak_age  ⟹  career_phase  0.0 → 1.0
#   Decline: peak_age → 42       ⟹  career_phase  1.0 → 0.0
def career_phase_value(age, peak_age, career_start=16, career_end=42):
    if age <= peak_age:
        return (age - career_start) / max(peak_age - career_start, 1)
    else:
        return max(0.0, 1.0 - (age - peak_age) / max(career_end - peak_age, 1))

players_data['career_phase'] = players_data.apply(
    lambda row: career_phase_value(
        row['age'],
        peak_age_map.get(row['best_position'], 28)
    ), axis=1
)

# ── Features & Labels ─────────────────────────────────────────────────────────
# Stable features only — won't change significantly over years
# User provides: name, age, position. Rest is auto-looked-up.
feature_cols = [
    'age', 'position_encoded', 'name_encoded', 'foot_encoded',
    'height_cm', 'potential',
    'weak_foot', 'skill_moves', 'international_reputation',
    'num_positions', 'career_phase',
]

output_cols = [
    'crossing', 'finishing', 'heading_accuracy', 'short_passing',
    'volleys', 'dribbling', 'curve', 'fk_accuracy', 'long_passing',
    'ball_control', 'acceleration', 'sprint_speed', 'agility',
    'reactions', 'balance', 'shot_power', 'jumping', 'stamina',
    'strength', 'long_shots', 'aggression', 'interceptions',
    'positioning', 'vision', 'penalties', 'composure',
    'defensive_awareness', 'standing_tackle', 'sliding_tackle',
    'gk_diving', 'gk_handling', 'gk_kicking', 'gk_positioning',
    'gk_reflexes', 'overall_rating', 'log_value'
]

players_data = players_data.dropna(subset=feature_cols + output_cols)

X = players_data[feature_cols]
y = players_data[output_cols]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── Model ─────────────────────────────────────────────────────────────────────
model = MultiOutputRegressor(XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=10,
    min_child_weight=2,
    subsample=0.85,
    colsample_bytree=0.9,
    reg_alpha=0.05,
    reg_lambda=0.5,
    random_state=42,
    n_jobs=-1,
    tree_method='hist',
))
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print(f"Model R² Score: {score:.4f} ({score*100:.2f}%)")

# ── Predict by name ──────────────────────────────────────────────────────────
# USER ONLY PROVIDES THESE 3 THINGS:
name = "Kylian Mbappé Lottin"
future_age = 29
position = "ST"  # None = keep current position, or set e.g. "RW", "ST"

# --- Auto-lookup everything else from the dataset ---
player = players_data[players_data['name'].str.contains(name, case=False, na=False)]
if player.empty:
    player = players_data[players_data['full_name'].str.contains(name, case=False, na=False)]
if player.empty:
    raise ValueError(f"Player '{name}' not found in dataset.")

# Handle multiple matches
if player['name_encoded'].nunique() > 1:
    print("Multiple players found:")
    print(player[['name', 'full_name', 'best_position', 'club_name']].drop_duplicates())
    raise ValueError("Be more specific with the name.")

# Use the most recent version
player = player.sort_values('version').iloc[-1]

# Determine position encoding
if position is not None:
    pos_encoded = le_position.transform([position])[0]
else:
    pos_encoded = player['position_encoded']

# Compute career_phase for the future age using the player's position
future_pos_name = position if position is not None else player['best_position']
future_career = career_phase_value(
    future_age,
    peak_age_map.get(future_pos_name, 28)
)

# Build prediction input — all extra features auto-filled from dataset
future_input = pd.DataFrame([{
    'age':                      future_age,
    'position_encoded':         pos_encoded,
    'name_encoded':             player['name_encoded'],
    'foot_encoded':             player['foot_encoded'],
    'height_cm':                player['height_cm'],
    'potential':                player['potential'],
    'weak_foot':                player['weak_foot'],
    'skill_moves':              player['skill_moves'],
    'international_reputation': player['international_reputation'],
    'num_positions':            player['num_positions'],
    'career_phase':             future_career,
}])

predicted = model.predict(future_input)[0]
result = pd.Series(predicted, index=output_cols)

# ── Post-Prediction Career Curve Adjustment ───────────────────────────────────
# Anchored on CURRENT overall & potential — grow until peak, then decline.
# Young players with high overall relative to age have HIGHER chance of
# reaching or surpassing their potential.
current_overall = float(player['overall_rating'])
current_potential = float(player['potential'])
current_age = float(player['age'])
peak_age = peak_age_map.get(future_pos_name, 28)

# Achievement ratio: how far ahead the player already is for their age
achievement_ratio = current_overall / max(current_potential, 1)
# Bonus for prodigies: if achievement_ratio > 0.85 at age < 22, they can exceed potential
prodigy_bonus = max(0, (achievement_ratio - 0.85) * 2) if current_age < 22 else 0
true_ceiling = current_potential + prodigy_bonus * 3  # can exceed potential by up to ~3-4 pts

if future_age <= peak_age:
    # ── GROWTH PHASE: linear interpolation current → true_ceiling ──
    if peak_age > current_age:
        progress = min((future_age - current_age) / (peak_age - current_age), 1.0)
    else:
        progress = 1.0
    target_overall = current_overall + progress * (true_ceiling - current_overall)
else:
    # ── DECLINE PHASE ──
    # Key insight: if a player is ALREADY past peak (current_age > peak_age),
    # their current overall IS their real baseline — they've already proven
    # they decline slowly. Don't re-apply the full peak→now decline.
    #
    # Decline rate: ~1-2 OVR per year past peak, accelerating slightly with age.
    # Max drop per year capped at 3 to prevent absurd collapses.
    if current_age >= peak_age:
        # Already past peak — decline only for the ADDITIONAL years
        years_to_decline = future_age - current_age
        baseline = current_overall
    else:
        # Currently pre-peak, predicting past peak
        years_to_decline = future_age - peak_age
        baseline = true_ceiling

    # Per-year decline rate: starts at ~1.0/yr, increases with age
    # At 30-33: ~1.0/yr, at 34-36: ~1.5/yr, at 37-39: ~2.0/yr, at 40+: ~2.5/yr
    total_decline = 0.0
    for yr in range(int(years_to_decline)):
        age_at_yr = (current_age if current_age >= peak_age else peak_age) + yr
        if age_at_yr < 33:
            rate = 1.0
        elif age_at_yr < 36:
            rate = 1.5
        elif age_at_yr < 39:
            rate = 2.0
        else:
            rate = 2.5
        total_decline += rate
    # Handle fractional year
    frac = years_to_decline - int(years_to_decline)
    if frac > 0:
        age_at_end = (current_age if current_age >= peak_age else peak_age) + int(years_to_decline)
        rate = 1.0 if age_at_end < 33 else (1.5 if age_at_end < 36 else (2.0 if age_at_end < 39 else 2.5))
        total_decline += rate * frac

    floor_rating = max(50, current_overall * 0.55)
    target_overall = max(floor_rating, baseline - total_decline)

# Scale ALL predicted stats proportionally based on career curve vs model prediction
model_overall = result.get('overall_rating', target_overall)
if model_overall > 0:
    scale = target_overall / model_overall
else:
    scale = 1.0

# Apply scale to all stats except log_value (market value scales differently)
for col in output_cols:
    if col == 'log_value':
        continue
    result[col] = result[col] * scale

# Explicitly set overall_rating to the career-curve target
result['overall_rating'] = target_overall


_ov = players_data['overall_rating'].values.astype(float)
_lv = players_data['log_value'].values.astype(float)
_value_coeffs = np.polyfit(_ov, _lv, 2)  # quadratic fit
adjusted_log_value = np.polyval(_value_coeffs, target_overall)


age_factor = 1.0
if future_age < peak_age:
    # Young = premium (up to ~15% extra)
    age_factor = 1.0 + 0.15 * (1.0 - future_age / peak_age)
elif future_age > peak_age:
    # Old = discount (up to ~40% off, gradual)
    years_past = future_age - peak_age
    age_factor = max(0.6, 1.0 - 0.03 * years_past)

# Apply age factor in real-value space, then convert back to log
adjusted_value = np.expm1(adjusted_log_value) * age_factor
adjusted_log_value = np.log1p(max(0, adjusted_value))

result['log_value'] = adjusted_log_value

print(f"\nCareer curve: peak_age={peak_age:.1f}, achievement={achievement_ratio:.3f}, "
      f"prodigy_bonus={prodigy_bonus:.2f}, ceiling={true_ceiling:.1f}")
print(f"Career-adjusted overall: {current_overall:.0f} → {target_overall:.1f} "
      f"(model raw: {model_overall:.1f}, scale: {scale:.3f})")

# Convert log_value back to actual market value
raw_value = np.expm1(result['log_value'])  # reverse log1p
if raw_value >= 1_000_000:
    value_str = f"€{raw_value / 1_000_000:.1f}M"
elif raw_value >= 1_000:
    value_str = f"€{raw_value / 1_000:.0f}K"
else:
    value_str = f"€{max(0, raw_value):.0f}"

# Show stats (integer) and value separately
predicted_stats = result.drop('log_value').round().astype(int)

display_name = clean_name(player['full_name'])

print(f"\nPredicted stats for {display_name} at age {future_age}:")
print(predicted_stats.to_string())
print(f"\nPredicted Market Value: {value_str}")

# ── Visualization ─────────────────────────────────────────────────────────────
from matplotlib.ticker import FuncFormatter

radar_cols = [
    'crossing', 'finishing', 'short_passing', 'dribbling',
    'ball_control', 'acceleration', 'sprint_speed',
    'reactions', 'shot_power', 'stamina', 'strength',
    'long_shots', 'vision', 'composure', 'positioning',
    'defensive_awareness', 'overall_rating'
]
radar_display = [
    'Crossing', 'Finishing', 'Short Pass', 'Dribbling',
    'Ball Ctrl', 'Accel', 'Sprint Spd',
    'Reactions', 'Shot Power', 'Stamina', 'Strength',
    'Long Shots', 'Vision', 'Composure', 'Positioning',
    'Def Aware', 'Overall'
]

current_stats = player[radar_cols].values.astype(float)
future_stats = predicted_stats[radar_cols].values.astype(float)

current_value = player['value_numeric']
predicted_value = np.expm1(result['log_value'])

def fmt_value(v):
    if v >= 1_000_000:
        return f"€{v / 1_000_000:.1f}M"
    elif v >= 1_000:
        return f"€{v / 1_000:.0f}K"
    return f"€{max(0, v):.0f}"


fig = plt.figure(figsize=(18, 14))
fig.suptitle(f"{display_name} — Age {int(player['age'])} → {future_age}",
             fontsize=16, fontweight='bold', color='#222', y=0.98)

# ── Player Photo Inset (top-center) ──────────────────────────────────────────
try:
    img_url = player.get('image', '')
    if img_url:
        req = Request(img_url, headers={'User-Agent': 'Mozilla/5.0'})
        img_data = urlopen(req, timeout=5).read()
        player_img = Image.open(BytesIO(img_data)).convert('RGBA')
        # Place photo as a small inset at top-center of figure
        ax_photo = fig.add_axes([0.455, 0.88, 0.09, 0.09])  # [left, bottom, width, height]
        ax_photo.imshow(player_img)
        ax_photo.axis('off')
except Exception as e:
    print(f"Could not load player image: {e}")

# --- Top-left: Radar Chart ---
ax1 = fig.add_subplot(221, polar=True)

num_vars = len(radar_cols)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

current_vals = current_stats.tolist() + [current_stats[0]]
future_vals = future_stats.tolist() + [future_stats[0]]

ax1.plot(angles, current_vals, 'o-', linewidth=2, markersize=4, color='#2196F3',
         label=f'Current (Age {int(player["age"])})')
ax1.fill(angles, current_vals, alpha=0.15, color='#2196F3')
ax1.plot(angles, future_vals, 'o-', linewidth=2, markersize=4, color='#FF5722',
         label=f'Predicted (Age {future_age})')
ax1.fill(angles, future_vals, alpha=0.15, color='#FF5722')

ax1.set_xticks(angles[:-1])
ax1.set_xticklabels(radar_display, size=7, color='#444', fontweight='bold')
ax1.set_ylim(0, 100)
ax1.set_yticks([20, 40, 60, 80, 100])
ax1.set_yticklabels(['20', '40', '60', '80', '100'], size=6, color='#888')
ax1.yaxis.grid(True, color='#ddd', linestyle='-', linewidth=0.5)
ax1.xaxis.grid(True, color='#ddd', linestyle='-', linewidth=0.5)
ax1.spines['polar'].set_color('#ccc')
ax1.set_title('Stats Comparison', size=12, fontweight='bold', pad=18, color='#333')
ax1.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=8)

# bargrph
ax2 = fig.add_subplot(222)

bars = ax2.bar(
    [f'Current\n(Age {int(player["age"])})', f'Predicted\n(Age {future_age})'],
    [current_value, predicted_value],
    color=['#2196F3', '#FF5722'],
    width=0.5, edgecolor='white', linewidth=1.5
)
for bar, val in zip(bars, [current_value, predicted_value]):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + max(current_value, predicted_value) * 0.02,
             fmt_value(val), ha='center', va='bottom', fontsize=12,
             fontweight='bold', color='#333')

ax2.set_ylabel('Market Value (€)', fontsize=10, color='#555')
ax2.set_title('Transfer Market Value', size=12, fontweight='bold', color='#333')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.tick_params(axis='x', labelsize=10)
ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: fmt_value(x)))
ax2.set_ylim(0, max(current_value, predicted_value) * 1.25)

# ── Bottom: Current Stats Table (left) & Predicted Stats Table (right) ────────
all_stat_cols = [c for c in output_cols if c != 'log_value']
current_stat_vals = [int(player[c]) for c in all_stat_cols]
future_stat_vals = [int(predicted_stats[c]) for c in all_stat_cols]
stat_labels = [c.replace('_', ' ').title() for c in all_stat_cols]

half = len(all_stat_cols) // 2 + 1

#letft table
ax3 = fig.add_subplot(223)
ax3.axis('off')
table1_data = [[stat_labels[i], current_stat_vals[i], future_stat_vals[i]]
               for i in range(half)]
table1 = ax3.table(
    cellText=table1_data,
    colLabels=['Stat', f'Current ({int(player["age"])})', f'Predicted ({future_age})'],
    cellLoc='center', loc='center',
    colColours=['#E3F2FD', '#BBDEFB', '#FFCCBC'],
)
table1.auto_set_font_size(False)
table1.set_fontsize(8)
table1.scale(1.0, 1.2)
for (row, col), cell in table1.get_celld().items():
    if row == 0:
        cell.set_text_props(fontweight='bold', color='#333')
    elif col == 2:
        cur = table1_data[row - 1][1]
        fut = table1_data[row - 1][2]
        if fut > cur:
            cell.set_facecolor('#C8E6C9')
        elif fut < cur:
            cell.set_facecolor('#FFCDD2')

# --- Bottom-right table ---
ax4 = fig.add_subplot(224)
ax4.axis('off')
table2_data = [[stat_labels[i], current_stat_vals[i], future_stat_vals[i]]
               for i in range(half, len(all_stat_cols))]
table2_data.append(['Market Value', fmt_value(current_value), fmt_value(predicted_value)])
table2 = ax4.table(
    cellText=table2_data,
    colLabels=['Stat', f'Current ({int(player["age"])})', f'Predicted ({future_age})'],
    cellLoc='center', loc='center',
    colColours=['#E3F2FD', '#BBDEFB', '#FFCCBC'],
)
table2.auto_set_font_size(False)
table2.set_fontsize(8)
table2.scale(1.0, 1.2)
for (row, col), cell in table2.get_celld().items():
    if row == 0:
        cell.set_text_props(fontweight='bold', color='#333')
    elif col == 2 and row - 1 < len(table2_data) - 1:
        idx = half + row - 1
        if idx < len(all_stat_cols):
            cur = current_stat_vals[idx]
            fut = future_stat_vals[idx]
            if fut > cur:
                cell.set_facecolor('#C8E6C9')
            elif fut < cur:
                cell.set_facecolor('#FFCDD2')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()