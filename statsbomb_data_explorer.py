"""
StatsBomb Data Explorer
-----------------------
Explore available football data with coordinates for heatmaps, shot maps, and passing networks.
Works with StatsBomb's free data (Bundesliga 2023/2024, Euro 2024, etc.)
"""
from xgboost import XGBRegressor
from statsbombpy import sb
import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from mplsoccer import Pitch, VerticalPitch
# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

def show_available_competitions():
    """Show all available competitions and seasons."""
    print("\n" + "="*70)
    print("AVAILABLE COMPETITIONS IN STATSBOMB FREE DATA")
    print("="*70)
    
    comps = sb.competitions()
    
    # Show unique combinations sorted by most recent
    display = comps[['competition_name', 'season_name', 'competition_id', 'season_id']].copy()
    display = display.sort_values('season_name', ascending=False)
    
    print(f"\nTotal competitions: {len(display)}")
    print("\n" + display.to_string(index=False))
    
    return comps


def select_competition():
    """Let user select a competition."""
    comps = show_available_competitions()
    
    print("\n" + "="*70)
    comp_name = input("Enter competition name (e.g., '1. Bundesliga', 'UEFA Euro', 'Premier League'): ").strip()
    season_name = input("Enter season name (e.g., '2023/2024', '2024', '2015/2016'): ").strip()
    
    # Try exact match first
    selected = comps[
        (comps['competition_name'] == comp_name) & 
        (comps['season_name'] == season_name)
    ]
    
    # Try fuzzy match if exact fails
    if selected.empty:
        selected = comps[
            (comps['competition_name'].str.contains(comp_name, case=False, na=False)) & 
            (comps['season_name'] == season_name)
        ]
    
    if selected.empty:
        print(f"\n No data found for '{comp_name}' {season_name}")
        print("\nDid you mean one of these?")
        matches = comps[comps['competition_name'].str.contains(comp_name, case=False, na=False)]
        if not matches.empty:
            print(matches[['competition_name', 'season_name']].drop_duplicates().to_string(index=False))
        return None, None
    
    if len(selected) > 1:
        print(f"\n⚠ Multiple matches found, using first one:")
        print(selected[['competition_name', 'season_name']].to_string(index=False))
    
    comp_id = selected['competition_id'].iloc[0]
    season_id = selected['season_id'].iloc[0]
    actual_comp = selected['competition_name'].iloc[0]
    
    print(f"\n Selected: {actual_comp} - {season_name}")
    print(f"  Competition ID: {comp_id}")
    print(f"  Season ID: {season_id}")
    
    return comp_id, season_id


def show_matches(comp_id, season_id):
    """Show all matches in the competition."""
    print("\n" + "="*70)
    print("AVAILABLE MATCHES")
    print("="*70)
    
    matches = sb.matches(competition_id=comp_id, season_id=season_id)
    
    print(f"\nTotal matches: {len(matches)}")
    print("\nSample matches:")
    display = matches[['match_id', 'match_date', 'home_team', 'away_team', 'home_score', 'away_score']].copy()
    
    for idx, row in display.head(20).iterrows():
        print(f"  {row['match_id']} | {row['match_date']} | "
              f"{row['home_team']} {row['home_score']}-{row['away_score']} {row['away_team']}")
    
    if len(matches) > 20:
        print(f"\n  ... and {len(matches) - 20} more matches")
    
    return matches


def show_players_in_match(match_id):
    """Show all players who played in a match with their event counts."""
    print("\n" + "="*70)
    print(f"PLAYERS IN MATCH {match_id}")
    print("="*70)
    
    events = sb.events(match_id=match_id)
    
    # Get player statistics
    player_stats = events.groupby(['player', 'team']).agg({
        'id': 'count',
        'type': lambda x: x.value_counts().to_dict()
    }).reset_index()
    player_stats.columns = ['player', 'team', 'total_events', 'event_types']
    
    # Sort by total events
    player_stats = player_stats.sort_values('total_events', ascending=False)
    
    print(f"\nTotal players: {len(player_stats)}")
    print("\n{:<40} {:<25} {:>10}".format("Player", "Team", "Events"))
    print("-"*78)
    
    for idx, row in player_stats.iterrows():
        player_name = row['player'][:38] if pd.notna(row['player']) else 'Unknown'
        team_name = row['team'][:23] if pd.notna(row['team']) else 'Unknown'
        print("{:<40} {:<25} {:>10}".format(player_name, team_name, row['total_events']))
    
    return events, player_stats


def get_player_events(events, player_name):
    """Get all events for a specific player with coordinates."""
    player_events = events[events['player'] == player_name].copy()
    
    if player_events.empty:
        print(f"\nNo events found for {player_name}")
        return None
    
    print("\n" + "="*70)
    print(f"EVENTS FOR {player_name}")
    print("="*70)
    print(f"Total events: {len(player_events)}")
    
    # Event type breakdown
    event_counts = player_events['type'].value_counts()
    print("\nEvent breakdown:")
    for event_type, count in event_counts.head(10).items():
        print(f"  {event_type:<20}: {count}")
    
    # Show events with coordinates
    coord_events = player_events[player_events['location'].notna()].copy()
    print(f"\nEvents with coordinates: {len(coord_events)}")
    
    # Extract x, y coordinates
    if len(coord_events) > 0:
        coord_events['x'] = coord_events['location'].apply(lambda loc: loc[0] if isinstance(loc, list) else None)
        coord_events['y'] = coord_events['location'].apply(lambda loc: loc[1] if isinstance(loc, list) else None)
        
        print("\nSample events with coordinates:")
        sample = coord_events[['minute', 'type', 'x', 'y']].head(10)
        print(sample.to_string(index=False))
    
    return player_events


def analyze_player_actions(player_events):
    """Analyze specific actions: passes, shots, touches."""
    if player_events is None or player_events.empty:
        return
    
    print("\n" + "="*70)
    print("DETAILED ACTION ANALYSIS")
    print("="*70)
    
    # Passes
    passes = player_events[player_events['type'] == 'Pass'].copy()
    if len(passes) > 0:
        passes['start_x'] = passes['location'].apply(lambda x: x[0] if isinstance(x, list) else None)
        passes['start_y'] = passes['location'].apply(lambda x: x[1] if isinstance(x, list) else None)
        passes['end_x'] = passes['pass_end_location'].apply(lambda x: x[0] if isinstance(x, list) and x else None)
        passes['end_y'] = passes['pass_end_location'].apply(lambda x: x[1] if isinstance(x, list) and x else None)
        
        print(f"\nPASSES: {len(passes)}")
        print("  Sample passes (start -> end):")
        for idx, row in passes[['start_x', 'start_y', 'end_x', 'end_y']].head(5).iterrows():
            if pd.notna(row['start_x']):
                print(f"    ({row['start_x']:.1f}, {row['start_y']:.1f}) -> ({row['end_x']:.1f}, {row['end_y']:.1f})")
    
    # Shots
    shots = player_events[player_events['type'] == 'Shot'].copy()
    if len(shots) > 0:
        shots['x'] = shots['location'].apply(lambda x: x[0] if isinstance(x, list) else None)
        shots['y'] = shots['location'].apply(lambda x: x[1] if isinstance(x, list) else None)
        
        print(f"\nSHOTS: {len(shots)}")
        print("  Shot locations:")
        for idx, row in shots[['x', 'y', 'shot_outcome']].iterrows():
            if pd.notna(row['x']):
                outcome = row['shot_outcome'] if pd.notna(row['shot_outcome']) else 'Unknown'
                print(f"    ({row['x']:.1f}, {row['y']:.1f}) - {outcome}")
    
    # All touches (any event with location)
    touches = player_events[player_events['location'].notna()].copy()
    touches['x'] = touches['location'].apply(lambda x: x[0] if isinstance(x, list) else None)
    touches['y'] = touches['location'].apply(lambda x: x[1] if isinstance(x, list) else None)
    
    print(f"\nALL TOUCHES: {len(touches)}")
    print(f"  Coordinates available for heatmap generation")
    
    return {
        'passes': passes,
        'shots': shots,
        'touches': touches
    }


# ═══════════════════════════════════════════════════════════════════
# MULTI-MATCH DATA COLLECTION
# ═══════════════════════════════════════════════════════════════════

def collect_player_data_from_multiple_matches(comp_id, season_id, player_name):
    """
    Collect data for a player across ALL matches in a season.
    Returns list of DataFrames (one per match) with pass/shot/touch data.
    """
    print("\n" + "="*70)
    print(f"COLLECTING DATA FOR {player_name} ACROSS ENTIRE SEASON")
    print("="*70)
    
    # Get all matches
    matches = sb.matches(competition_id=comp_id, season_id=season_id)
    
    player_match_data = []
    
    for idx, match_row in matches.iterrows():
        match_id = match_row['match_id']
        try:
            # Get events for this match
            events = sb.events(match_id=match_id)
            
            # Filter for this player
            player_events = events[events['player'] == player_name].copy()
            
            if len(player_events) > 0:
                # Extract passes
                passes = player_events[player_events['type'] == 'Pass'].copy()
                if len(passes) > 0:
                    passes['start_x'] = passes['location'].apply(lambda x: x[0] if isinstance(x, list) else None)
                    passes['start_y'] = passes['location'].apply(lambda x: x[1] if isinstance(x, list) else None)
                    passes['end_x'] = passes['pass_end_location'].apply(lambda x: x[0] if isinstance(x, list) and x else None)
                    passes['end_y'] = passes['pass_end_location'].apply(lambda x: x[1] if isinstance(x, list) and x else None)
                
                # Extract shots
                shots = player_events[player_events['type'] == 'Shot'].copy()
                if len(shots) > 0:
                    shots['x'] = shots['location'].apply(lambda x: x[0] if isinstance(x, list) else None)
                    shots['y'] = shots['location'].apply(lambda x: x[1] if isinstance(x, list) else None)
                    # Keep shot_outcome column if it exists
                    if 'shot_outcome' not in shots.columns:
                        shots['shot_outcome'] = 'Unknown'
                
                # Extract touches
                touches = player_events[player_events['location'].notna()].copy()
                touches['x'] = touches['location'].apply(lambda x: x[0] if isinstance(x, list) else None)
                touches['y'] = touches['location'].apply(lambda x: x[1] if isinstance(x, list) else None)
                
                player_match_data.append({
                    'match_id': match_id,
                    'match_date': match_row['match_date'],
                    'passes': passes,
                    'shots': shots,
                    'touches': touches
                })
                
                print(f"  Match {match_id}: {len(passes)} passes, {len(shots)} shots, {len(touches)} touches")
        
        except Exception as e:
            # Skip matches where data isn't available
            continue
    
    print(f"\nCollected data from {len(player_match_data)} matches")
    return player_match_data


# ═══════════════════════════════════════════════════════════════════
# PREDICTION MODELS (Using Multi-Match Data)
# ═══════════════════════════════════════════════════════════════════

def train_pass_prediction_model(player_match_data):
    """
    Train XGBoost model to predict pass locations.
    Uses last 5 matches to predict next match.
    """
    if len(player_match_data) < 6:
        print(f"\n⚠ Need at least 6 matches, only have {len(player_match_data)}")
        return None
    
    print("\n" + "="*70)
    print("TRAINING PASS PREDICTION MODEL")
    print("="*70)
    
    X = []
    y = []
    
    # Create sliding windows across matches
    for i in range(len(player_match_data) - 5):
        past_5 = player_match_data[i:i+5]
        next_match = player_match_data[i+5]
        
        # Aggregate features from past 5 matches
        all_passes = pd.concat([m['passes'] for m in past_5], ignore_index=True)
        
        if len(all_passes) > 0 and len(next_match['passes']) > 0:
            features = [
                all_passes['start_x'].mean(),
                all_passes['start_y'].mean(),
                all_passes['end_x'].mean(),
                all_passes['end_y'].mean(),
                all_passes['start_x'].std(),
                all_passes['start_y'].std(),
                len(all_passes)  # Total pass count
            ]
            
            # Target: Average pass locations in next match
            labels = [
                next_match['passes']['start_x'].mean(),
                next_match['passes']['start_y'].mean(),
                next_match['passes']['end_x'].mean(),
                next_match['passes']['end_y'].mean()
            ]
            
            X.append(features)
            y.append(labels)
    
    if len(X) == 0:
        print("⚠ Not enough data to train")
        return None
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Training samples: {len(X)}")
    print(f"Features: {X.shape[1]}, Outputs: {y.shape[1]}")
    
    # Train model
    model = MultiOutputRegressor(XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1))
    model.fit(X, y)
    
    # Predict on last window
    y_pred = model.predict(X[-1:])
    print(f"\nPredicted pass zones for next match:")
    print(f"  Start: ({y_pred[0][0]:.1f}, {y_pred[0][1]:.1f})")
    print(f"  End: ({y_pred[0][2]:.1f}, {y_pred[0][3]:.1f})")
    
    return model


def train_shot_prediction_model(player_match_data):
    """
    Train XGBoost model to predict shot locations.
    """
    if len(player_match_data) < 6:
        print(f"\n⚠ Need at least 6 matches for shot model")
        return None
    
    print("\n" + "="*70)
    print("TRAINING SHOT PREDICTION MODEL")
    print("="*70)
    
    X = []
    y = []
    
    for i in range(len(player_match_data) - 5):
        past_5 = player_match_data[i:i+5]
        next_match = player_match_data[i+5]
        
        all_shots = pd.concat([m['shots'] for m in past_5 if len(m['shots']) > 0], ignore_index=True)
        
        if len(all_shots) > 0 and len(next_match['shots']) > 0:
            features = [
                all_shots['x'].mean(),
                all_shots['y'].mean(),
                all_shots['x'].std(),
                all_shots['y'].std(),
                len(all_shots)
            ]
            
            labels = [
                next_match['shots']['x'].mean(),
                next_match['shots']['y'].mean()
            ]
            
            X.append(features)
            y.append(labels)
    
    if len(X) == 0:
        print("⚠ Not enough shot data to train")
        return None
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Training samples: {len(X)}")
    
    model = MultiOutputRegressor(XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1))
    model.fit(X, y)
    
    y_pred = model.predict(X[-1:])
    print(f"\nPredicted shot zone: ({y_pred[0][0]:.1f}, {y_pred[0][1]:.1f})")
    
    return model


def train_touch_heatmap_model(player_match_data):
    """
    Train XGBoost model to predict general touch/heatmap zones.
    """
    if len(player_match_data) < 6:
        print(f"\n⚠ Need at least 6 matches for heatmap model")
        return None
    
    print("\n" + "="*70)
    print("TRAINING TOUCH/HEATMAP MODEL")
    print("="*70)
    
    X = []
    y = []
    
    for i in range(len(player_match_data) - 5):
        past_5 = player_match_data[i:i+5]
        next_match = player_match_data[i+5]
        
        all_touches = pd.concat([m['touches'] for m in past_5], ignore_index=True)
        
        if len(all_touches) > 0 and len(next_match['touches']) > 0:
            features = [
                all_touches['x'].mean(),
                all_touches['y'].mean(),
                all_touches['x'].std(),
                all_touches['y'].std(),
                all_touches['x'].quantile(0.25),
                all_touches['x'].quantile(0.75),
                len(all_touches)
            ]
            
            labels = [
                next_match['touches']['x'].mean(),
                next_match['touches']['y'].mean()
            ]
            
            X.append(features)
            y.append(labels)
    
    if len(X) == 0:
        print("⚠ Not enough touch data")
        return None
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Training samples: {len(X)}")
    
    model = MultiOutputRegressor(XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1))
    model.fit(X, y)
    
    y_pred = model.predict(X[-1:])
    print(f"\nPredicted heatmap center: ({y_pred[0][0]:.1f}, {y_pred[0][1]:.1f})")
    
    return model


# ═══════════════════════════════════════════════════════════════════
# VISUALIZATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def visualize_predictions(player_match_data, pass_model, shot_model, heatmap_model, player_name):
    """
    Visualize predictions for next match on a pitch.
    Shows predicted heatmap, passes, and shots.
    """
    print("\n" + "="*70)
    print("GENERATING PREDICTION VISUALIZATIONS")
    print("="*70)
    
    # Get last 5 matches for prediction input
    past_5 = player_match_data[-5:]
    
    # Prepare features for each model
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'{player_name} - Next Match Predictions', fontsize=16, fontweight='bold')
    
    # 1. HEATMAP PREDICTION
    if heatmap_model:
        ax = axes[0]
        pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='white')
        pitch.draw(ax=ax)
        ax.set_title('Predicted Touch Heatmap', fontsize=12, fontweight='bold')
        
        # Get last 5 matches touches
        all_touches = pd.concat([m['touches'] for m in past_5], ignore_index=True)
        
        if len(all_touches) > 0:
            features = np.array([[
                all_touches['x'].mean(),
                all_touches['y'].mean(),
                all_touches['x'].std(),
                all_touches['y'].std(),
                all_touches['x'].quantile(0.25),
                all_touches['x'].quantile(0.75),
                len(all_touches)
            ]])
            
            pred = heatmap_model.predict(features)
            
            # Plot prediction as heatmap center
            pitch.scatter(pred[0][0], pred[0][1], s=500, c='red', marker='*', 
                         edgecolors='yellow', linewidths=2, ax=ax, zorder=3, 
                         label='Predicted Center')
            
            # Show historical touches as context
            pitch.kdeplot(all_touches['x'], all_touches['y'], ax=ax, 
                         cmap='hot', shade=True, alpha=0.5, zorder=1)
            
            ax.legend(loc='upper left', fontsize=8)
    
    # 2. PASS PREDICTION
    if pass_model:
        ax = axes[1]
        pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='white')
        pitch.draw(ax=ax)
        ax.set_title('Predicted Pass Zones', fontsize=12, fontweight='bold')
        
        all_passes = pd.concat([m['passes'] for m in past_5], ignore_index=True)
        
        if len(all_passes) > 0:
            features = np.array([[
                all_passes['start_x'].mean(),
                all_passes['start_y'].mean(),
                all_passes['end_x'].mean(),
                all_passes['end_y'].mean(),
                all_passes['start_x'].std(),
                all_passes['start_y'].std(),
                len(all_passes)
            ]])
            
            pred = pass_model.predict(features)
            
            # Draw predicted pass arrow
            pitch.arrows(pred[0][0], pred[0][1], pred[0][2], pred[0][3],
                        width=3, headwidth=8, headlength=8, color='cyan', 
                        ax=ax, zorder=3, label='Predicted Pass Zone')
            
            # Show historical passes
            pitch.lines(all_passes['start_x'].head(30), all_passes['start_y'].head(30),
                       all_passes['end_x'].head(30), all_passes['end_y'].head(30),
                       ax=ax, color='white', alpha=0.3, zorder=1, lw=1)
            
            ax.legend(loc='upper left', fontsize=8)
    
    # 3. SHOT PREDICTION
    if shot_model:
        ax = axes[2]
        pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='white')
        pitch.draw(ax=ax)
        ax.set_title('Predicted Shot Zones', fontsize=12, fontweight='bold')
        
        all_shots = pd.concat([m['shots'] for m in past_5 if len(m['shots']) > 0], ignore_index=True)
        
        if len(all_shots) > 0:
            features = np.array([[
                all_shots['x'].mean(),
                all_shots['y'].mean(),
                all_shots['x'].std(),
                all_shots['y'].std(),
                len(all_shots)
            ]])
            
            pred = shot_model.predict(features)
            
            # Plot predicted shot zone
            pitch.scatter(pred[0][0], pred[0][1], s=800, marker='football', 
                         c='yellow', edgecolors='red', linewidths=2, ax=ax, zorder=3,
                         label='Predicted Shot Zone')
            
            # Show historical shots
            pitch.scatter(all_shots['x'], all_shots['y'], s=100, 
                         c='white', alpha=0.5, ax=ax, zorder=1, label='Past 5 Matches')
            
            ax.legend(loc='upper left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'predictions_{player_name.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: predictions_{player_name.replace(' ', '_')}.png")
    plt.show()


def plot_performance_trends(player_match_data, player_name):
    """
    Create line graph comparing metrics from last 10 matches:
    - Average heatmap position
    - Number of passes
    - Number of shots
    - Shots on target
    """
    print("\n" + "="*70)
    print("GENERATING PERFORMANCE TREND ANALYSIS")
    print("="*70)
    
    # Take last 10 matches (or all if less)
    matches_to_plot = player_match_data[-10:]
    
    metrics = {
        'match_num': [],
        'passes': [],
        'shots': [],
        'shots_on_target': [],
        'avg_x_position': [],
        'avg_y_position': []
    }
    
    for idx, match in enumerate(matches_to_plot, 1):
        metrics['match_num'].append(idx)
        metrics['passes'].append(len(match['passes']))
        
        # Count shots
        shots_df = match['shots']
        metrics['shots'].append(len(shots_df))
        
        # Count shots on target
        if len(shots_df) > 0 and 'shot_outcome' in shots_df.columns:
            on_target = shots_df[shots_df['shot_outcome'].isin(['Goal', 'Saved'])].shape[0]
        else:
            on_target = 0
        metrics['shots_on_target'].append(on_target)
        
        # Average position from touches
        touches = match['touches']
        if len(touches) > 0:
            metrics['avg_x_position'].append(touches['x'].mean())
            metrics['avg_y_position'].append(touches['y'].mean())
        else:
            metrics['avg_x_position'].append(0)
            metrics['avg_y_position'].append(0)
    
    df_metrics = pd.DataFrame(metrics)
    
    # Calculate 10-match averages
    avg_passes = df_metrics['passes'].mean()
    avg_shots = df_metrics['shots'].mean()
    avg_sot = df_metrics['shots_on_target'].mean()
    avg_x = df_metrics['avg_x_position'].mean()
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{player_name} - Last {len(matches_to_plot)} Matches Performance Trends', 
                 fontsize=16, fontweight='bold')
    
    # 1. Passes
    ax = axes[0, 0]
    ax.plot(df_metrics['match_num'], df_metrics['passes'], marker='o', 
            linewidth=2, markersize=8, color='#1f77b4', label='Passes')
    ax.axhline(y=avg_passes, color='red', linestyle='--', linewidth=2, 
               label=f'Avg: {avg_passes:.1f}')
    ax.set_title('Passes per Match', fontsize=12, fontweight='bold')
    ax.set_xlabel('Match Number')
    ax.set_ylabel('Number of Passes')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Shots
    ax = axes[0, 1]
    ax.plot(df_metrics['match_num'], df_metrics['shots'], marker='s', 
            linewidth=2, markersize=8, color='#ff7f0e', label='Total Shots')
    ax.plot(df_metrics['match_num'], df_metrics['shots_on_target'], marker='^', 
            linewidth=2, markersize=8, color='#2ca02c', label='Shots on Target')
    ax.axhline(y=avg_shots, color='#ff7f0e', linestyle='--', linewidth=1.5, alpha=0.7,
               label=f'Avg Shots: {avg_shots:.1f}')
    ax.axhline(y=avg_sot, color='#2ca02c', linestyle='--', linewidth=1.5, alpha=0.7,
               label=f'Avg SoT: {avg_sot:.1f}')
    ax.set_title('Shots Analysis', fontsize=12, fontweight='bold')
    ax.set_xlabel('Match Number')
    ax.set_ylabel('Number of Shots')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Average X Position (field position)
    ax = axes[1, 0]
    ax.plot(df_metrics['match_num'], df_metrics['avg_x_position'], marker='D', 
            linewidth=2, markersize=8, color='#d62728', label='Avg X Position')
    ax.axhline(y=avg_x, color='purple', linestyle='--', linewidth=2, 
               label=f'Avg: {avg_x:.1f}')
    ax.set_title('Average Field Position (X-axis)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Match Number')
    ax.set_ylabel('Average X Position (0=own goal, 120=opp goal)')
    ax.set_ylim(0, 120)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Summary Table
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_data = [
        ['Metric', 'Last Match', '10-Match Avg', 'Trend'],
        ['Passes', f"{df_metrics['passes'].iloc[-1]}", f"{avg_passes:.1f}", 
         '↑' if df_metrics['passes'].iloc[-1] > avg_passes else '↓'],
        ['Shots', f"{df_metrics['shots'].iloc[-1]}", f"{avg_shots:.1f}",
         '↑' if df_metrics['shots'].iloc[-1] > avg_shots else '↓'],
        ['Shots on Target', f"{df_metrics['shots_on_target'].iloc[-1]}", f"{avg_sot:.1f}",
         '↑' if df_metrics['shots_on_target'].iloc[-1] > avg_sot else '↓'],
        ['Avg Position', f"{df_metrics['avg_x_position'].iloc[-1]:.1f}", f"{avg_x:.1f}",
         '↑' if df_metrics['avg_x_position'].iloc[-1] > avg_x else '↓']
    ]
    
    table = ax.table(cellText=summary_data, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.2, 0.25, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style data rows
    for i in range(1, 5):
        for j in range(4):
            table[(i, j)].set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
    
    ax.set_title('Performance Summary', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(f'trends_{player_name.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: trends_{player_name.replace(' ', '_')}.png")
    plt.show()
    
    return df_metrics
# ═══════════════════════════════════════════════════════════════════
# MAIN WORKFLOW
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*70)
    print("STATSBOMB DATA EXPLORER & COORDINATE PREDICTION")
    print("="*70)
    print("This tool can:")
    print("  1. Explore single match data (for visualization)")
    print("  2. Train prediction models (requires multiple matches)")
    
    mode = input("\nEnter mode (1=single match, 2=train models): ").strip()
    
    # Step 1: Select competition
    comp_id, season_id = select_competition()
    if comp_id is None:
        exit()
    
    if mode == "1":
        # ═══════════════════════════════════════════════════════════
        # MODE 1: Single Match Exploration
        # ═══════════════════════════════════════════════════════════
        matches = show_matches(comp_id, season_id)
        
        print("\n" + "="*70)
        match_id = int(input("Enter match ID to explore: ").strip())
        
        events, player_stats = show_players_in_match(match_id)
        
        print("\n" + "="*70)
        player_name = input("Enter exact player name: ").strip()
        
        player_events = get_player_events(events, player_name)
        
        if player_events is not None:
            action_data = analyze_player_actions(player_events)
            
            print("\n" + "="*70)
            print("DATA READY FOR VISUALIZATION")
            print("="*70)
            print("You can use these DataFrames to create visualizations:")
            print(f"  - action_data['passes']: {len(action_data['passes'])} passes")
            print(f"  - action_data['shots']: {len(action_data['shots'])} shots")
            print(f"  - action_data['touches']: {len(action_data['touches'])} touches")
            print("\nCoordinate columns:")
            print("  Passes: start_x, start_y, end_x, end_y")
            print("  Shots/Touches: x, y")
            print("  Pitch dimensions: 120x80 (StatsBomb format)")
    
    elif mode == "2":
        # ═══════════════════════════════════════════════════════════
        # MODE 2: Multi-Match Training
        # ═══════════════════════════════════════════════════════════
        print("\n" + "="*70)
        player_name = input("Enter player name to train models for: ").strip()
        
        # Collect data from all matches
        player_match_data = collect_player_data_from_multiple_matches(comp_id, season_id, player_name)
        
        if len(player_match_data) < 6:
            print(f"\n⚠ ERROR: Need at least 6 matches, found only {len(player_match_data)}")
            print("Try a different player or competition with more matches")
            exit()
        
        # Train three separate models
        print("\n" + "="*70)
        print("TRAINING ALL MODELS")
        print("="*70)
        
        pass_model = train_pass_prediction_model(player_match_data)
        shot_model = train_shot_prediction_model(player_match_data)
        heatmap_model = train_touch_heatmap_model(player_match_data)
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print("Models trained:")
        print(f"  ✓ Pass prediction model: {'Ready' if pass_model else 'Failed'}")
        print(f"  ✓ Shot prediction model: {'Ready' if shot_model else 'Failed'}")
        print(f"  ✓ Heatmap prediction model: {'Ready' if heatmap_model else 'Failed'}")
        
        # Generate visualizations
        if pass_model or shot_model or heatmap_model:
            print("\n" + "="*70)
            print("GENERATING VISUALIZATIONS")
            print("="*70)
            
            # 1. Prediction visualizations
            visualize_predictions(player_match_data, pass_model, shot_model, 
                                 heatmap_model, player_name)
            
            # 2. Performance trends
            metrics_df = plot_performance_trends(player_match_data, player_name)
            
            print("\n" + "="*70)
            print("ANALYSIS COMPLETE")
            print("="*70)
            print("Generated files:")
            print(f"  ✓ predictions_{player_name.replace(' ', '_')}.png - Next match predictions")
            print(f"  ✓ trends_{player_name.replace(' ', '_')}.png - Last 10 matches analysis")
            print("\nPredictions are based on the player's last 5 matches")
            print("Trends show performance over the last 10 matches")
    
    else:
        print("Invalid mode selected")
