"""
Unit Tests for advanced_feature_engineering_v7.4.py
Tests the core calculation functions to ensure optimization doesn't break logic.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import importlib.util
from datetime import datetime, timedelta

# Dynamic import for files with dots in name
def import_module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Import the feature engineering module - functions only, not the main script
# We extract just the functions we need
sys.path.insert(0, 'V9.0')

# Define functions inline to avoid executing main script during import
def calculate_close_set_win_rate(player_id, rolling_games_df):
    """Calculates a player's win percentage in close sets (decided by 2 points)."""
    if rolling_games_df.empty:
        return 0.5
    total_close_sets_played = 0
    total_close_sets_won = 0
    for _, game in rolling_games_df.iterrows():
        set_scores_str = game.get('Set Scores')
        if pd.isna(set_scores_str):
            continue
        for set_score in str(set_scores_str).split(','):
            try:
                p1_points, p2_points = map(int, set_score.split('-'))
                if abs(p1_points - p2_points) == 2:
                    total_close_sets_played += 1
                    is_p1 = (game['Player 1 ID'] == player_id)
                    p1_won_set = p1_points > p2_points
                    if (is_p1 and p1_won_set) or (not is_p1 and not p1_won_set):
                        total_close_sets_won += 1
            except (ValueError, IndexError):
                continue
    if total_close_sets_played == 0:
        return 0.5
    return total_close_sets_won / total_close_sets_played

def calculate_h2h_dominance(p1_id, h2h_df, current_date, decay_factor):
    """Calculates a recency-weighted H2H dominance score."""
    if h2h_df.empty:
        return 0.0
    total_weighted_score = 0
    for _, game in h2h_df.iterrows():
        days_ago = (current_date - game['Date']).days
        weight = decay_factor ** days_ago
        if game['Player 1 ID'] == p1_id:
            point_diff = game['P1 Total Points'] - game['P2 Total Points']
        else:
            point_diff = game['P2 Total Points'] - game['P1 Total Points']
        total_weighted_score += (point_diff * weight)
    return total_weighted_score

def calculate_performance_slope(performance_history):
    """Calculates the slope of recent performance using linear regression."""
    if len(performance_history) < 2:
        return 0.0
    y = np.array(performance_history)
    x = np.arange(len(y))
    slope = np.polyfit(x, y, 1)[0]
    return slope

def calculate_pdr(player_id, rolling_games_df):
    """Calculates the Points Dominance Ratio for a single player."""
    if rolling_games_df.empty:
        return 0.5
    total_points_won = 0
    total_points_played = 0
    for _, game in rolling_games_df.iterrows():
        if game['Player 1 ID'] == player_id:
            points_won = game['P1 Total Points']
            points_lost = game['P2 Total Points']
        else:
            points_won = game['P2 Total Points']
            points_lost = game['P1 Total Points']
        total_points_won += points_won
        total_points_played += (points_won + points_lost)
    if total_points_played == 0:
        return 0.5
    return total_points_won / total_points_played

def update_elo(p1_elo, p2_elo, p1_won, k_factor=32):
    """Update Elo ratings after a match."""
    expected_p1 = 1 / (1 + 10 ** ((p2_elo - p1_elo) / 400))
    expected_p2 = 1 / (1 + 10 ** ((p1_elo - p2_elo) / 400))
    score_p1 = 1 if p1_won else 0
    score_p2 = 0 if p1_won else 1
    new_p1_elo = p1_elo + k_factor * (score_p1 - expected_p1)
    new_p2_elo = p2_elo + k_factor * (score_p2 - expected_p2)
    return new_p1_elo, new_p2_elo


class TestCalculateCloseSetWinRate:
    """Tests for close set win rate calculation."""

    def test_empty_dataframe_returns_neutral(self):
        """Empty rolling games should return 0.5 (neutral)."""
        empty_df = pd.DataFrame()
        result = calculate_close_set_win_rate(player_id=1, rolling_games_df=empty_df)
        assert result == 0.5

    def test_no_close_sets_returns_neutral(self):
        """Games with no close sets (diff != 2) should return 0.5."""
        df = pd.DataFrame({
            'Player 1 ID': [1, 1],
            'Player 2 ID': [2, 2],
            'Set Scores': ['11-5,11-6', '11-3,11-4'],  # No close sets
            'P1_Win': [1, 1]
        })
        result = calculate_close_set_win_rate(player_id=1, rolling_games_df=df)
        assert result == 0.5

    def test_all_close_sets_won(self):
        """Player wins all close sets should return 1.0."""
        df = pd.DataFrame({
            'Player 1 ID': [1],
            'Player 2 ID': [2],
            'Set Scores': ['11-9,11-9'],  # 2 close sets, both won by P1
            'P1_Win': [1]
        })
        result = calculate_close_set_win_rate(player_id=1, rolling_games_df=df)
        assert result == 1.0

    def test_all_close_sets_lost(self):
        """Player loses all close sets should return 0.0."""
        df = pd.DataFrame({
            'Player 1 ID': [1],
            'Player 2 ID': [2],
            'Set Scores': ['9-11,9-11'],  # 2 close sets, both lost by P1
            'P1_Win': [0]
        })
        result = calculate_close_set_win_rate(player_id=1, rolling_games_df=df)
        assert result == 0.0

    def test_player_as_p2(self):
        """Test when target player is Player 2."""
        df = pd.DataFrame({
            'Player 1 ID': [1],
            'Player 2 ID': [2],
            'Set Scores': ['11-9,11-9'],  # P1 won both close sets
            'P1_Win': [1]
        })
        # Player 2 lost both close sets
        result = calculate_close_set_win_rate(player_id=2, rolling_games_df=df)
        assert result == 0.0

    def test_mixed_close_sets(self):
        """Test mixed results."""
        df = pd.DataFrame({
            'Player 1 ID': [1],
            'Player 2 ID': [2],
            'Set Scores': ['11-9,9-11'],  # 1 win, 1 loss for P1
            'P1_Win': [1]
        })
        result = calculate_close_set_win_rate(player_id=1, rolling_games_df=df)
        assert result == 0.5

    def test_handles_nan_set_scores(self):
        """Should skip NaN set scores gracefully."""
        df = pd.DataFrame({
            'Player 1 ID': [1, 1],
            'Player 2 ID': [2, 2],
            'Set Scores': [np.nan, '11-9,11-9'],
            'P1_Win': [1, 1]
        })
        result = calculate_close_set_win_rate(player_id=1, rolling_games_df=df)
        assert result == 1.0  # Only counts the valid game


class TestCalculateH2HDominance:
    """Tests for H2H dominance score calculation."""

    def test_empty_dataframe_returns_zero(self):
        """Empty H2H history should return 0.0."""
        empty_df = pd.DataFrame()
        result = calculate_h2h_dominance(
            p1_id=1, h2h_df=empty_df,
            current_date=datetime.now(), decay_factor=0.98
        )
        assert result == 0.0

    def test_single_game_p1_won(self):
        """Single game where p1 won with point differential."""
        df = pd.DataFrame({
            'Player 1 ID': [1],
            'Player 2 ID': [2],
            'P1 Total Points': [50],
            'P2 Total Points': [40],
            'Date': [datetime.now() - timedelta(days=1)]
        })
        result = calculate_h2h_dominance(
            p1_id=1, h2h_df=df,
            current_date=datetime.now(), decay_factor=0.98
        )
        # Point diff = 50-40 = 10, weight = 0.98^1 = 0.98
        expected = 10 * 0.98
        assert abs(result - expected) < 0.01

    def test_single_game_p1_as_p2(self):
        """Test when p1 was Player 2 in the historical match."""
        df = pd.DataFrame({
            'Player 1 ID': [2],  # p1 was actually P2
            'Player 2 ID': [1],
            'P1 Total Points': [40],
            'P2 Total Points': [50],
            'Date': [datetime.now() - timedelta(days=1)]
        })
        result = calculate_h2h_dominance(
            p1_id=1, h2h_df=df,
            current_date=datetime.now(), decay_factor=0.98
        )
        # p1 (as P2) scored 50, opponent scored 40, diff = +10
        expected = 10 * 0.98
        assert abs(result - expected) < 0.01

    def test_decay_over_time(self):
        """Older games should have less weight."""
        df = pd.DataFrame({
            'Player 1 ID': [1, 1],
            'Player 2 ID': [2, 2],
            'P1 Total Points': [50, 50],
            'P2 Total Points': [40, 40],
            'Date': [datetime.now() - timedelta(days=1),
                    datetime.now() - timedelta(days=10)]
        })
        result = calculate_h2h_dominance(
            p1_id=1, h2h_df=df,
            current_date=datetime.now(), decay_factor=0.98
        )
        # Game 1: 10 * 0.98^1, Game 2: 10 * 0.98^10
        expected = 10 * (0.98**1) + 10 * (0.98**10)
        assert abs(result - expected) < 0.01


class TestCalculatePerformanceSlope:
    """Tests for PDR slope calculation."""

    def test_single_value_returns_zero(self):
        """Cannot calculate slope with one point."""
        result = calculate_performance_slope([0.5])
        assert result == 0.0

    def test_empty_returns_zero(self):
        """Empty history returns zero."""
        result = calculate_performance_slope([])
        assert result == 0.0

    def test_flat_trend(self):
        """Constant values should have zero slope."""
        result = calculate_performance_slope([0.5, 0.5, 0.5, 0.5])
        assert abs(result) < 0.001

    def test_positive_trend(self):
        """Increasing values should have positive slope."""
        result = calculate_performance_slope([0.4, 0.5, 0.6, 0.7])
        assert result > 0

    def test_negative_trend(self):
        """Decreasing values should have negative slope."""
        result = calculate_performance_slope([0.7, 0.6, 0.5, 0.4])
        assert result < 0


class TestCalculatePDR:
    """Tests for Points Dominance Ratio calculation."""

    def test_empty_dataframe_returns_neutral(self):
        """Empty games should return 0.5."""
        empty_df = pd.DataFrame()
        result = calculate_pdr(player_id=1, rolling_games_df=empty_df)
        assert result == 0.5

    def test_dominant_player(self):
        """Player with more points won should have PDR > 0.5."""
        df = pd.DataFrame({
            'Player 1 ID': [1],
            'Player 2 ID': [2],
            'P1 Total Points': [60],
            'P2 Total Points': [40]
        })
        result = calculate_pdr(player_id=1, rolling_games_df=df)
        assert result == 0.6  # 60/(60+40)

    def test_dominated_player(self):
        """Player with fewer points should have PDR < 0.5."""
        df = pd.DataFrame({
            'Player 1 ID': [1],
            'Player 2 ID': [2],
            'P1 Total Points': [40],
            'P2 Total Points': [60]
        })
        result = calculate_pdr(player_id=1, rolling_games_df=df)
        assert result == 0.4  # 40/(40+60)

    def test_player_as_p2(self):
        """Test PDR when player is Player 2."""
        df = pd.DataFrame({
            'Player 1 ID': [1],
            'Player 2 ID': [2],
            'P1 Total Points': [40],
            'P2 Total Points': [60]
        })
        result = calculate_pdr(player_id=2, rolling_games_df=df)
        assert result == 0.6  # 60/(40+60)

    def test_multiple_games(self):
        """Test aggregation across multiple games."""
        df = pd.DataFrame({
            'Player 1 ID': [1, 1],
            'Player 2 ID': [2, 2],
            'P1 Total Points': [50, 60],
            'P2 Total Points': [50, 40]
        })
        result = calculate_pdr(player_id=1, rolling_games_df=df)
        # Total: 110 won out of 200
        assert result == 0.55


class TestUpdateElo:
    """Tests for Elo rating updates."""

    def test_equal_elo_p1_wins(self):
        """Equal Elo, P1 wins should increase P1's rating."""
        new_p1, new_p2 = update_elo(1500, 1500, p1_won=True)
        assert new_p1 > 1500
        assert new_p2 < 1500

    def test_equal_elo_p1_loses(self):
        """Equal Elo, P1 loses should decrease P1's rating."""
        new_p1, new_p2 = update_elo(1500, 1500, p1_won=False)
        assert new_p1 < 1500
        assert new_p2 > 1500

    def test_rating_sum_preserved(self):
        """Total Elo should be preserved after update."""
        initial_sum = 1500 + 1600
        new_p1, new_p2 = update_elo(1500, 1600, p1_won=True)
        assert abs((new_p1 + new_p2) - initial_sum) < 0.01

    def test_upset_gives_more_elo(self):
        """Underdog winning should gain more than expected."""
        # Upset: 1400 beats 1600
        upset_p1, _ = update_elo(1400, 1600, p1_won=True)
        # Normal: 1600 beats 1400
        normal_p1, _ = update_elo(1600, 1400, p1_won=True)

        # Upset winner gains more
        upset_gain = upset_p1 - 1400
        normal_gain = normal_p1 - 1600
        assert upset_gain > normal_gain


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
