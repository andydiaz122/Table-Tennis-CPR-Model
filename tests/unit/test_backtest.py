"""
Unit Tests for backtest_final_v7.4.py
Tests helper functions used in backtesting to ensure optimization doesn't break logic.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# Functions copied from backtest for isolated testing
def calculate_close_set_win_rate(player_id, rolling_games_df):
    if rolling_games_df.empty:
        return 0.5
    total_close_sets_played, total_close_sets_won = 0, 0
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


def calculate_h2h_dominance(p1_id, h2h_df, current_match_date, decay_factor):
    if h2h_df.empty:
        return 0.0
    total_weighted_score = 0
    for _, game in h2h_df.iterrows():
        days_ago = (current_match_date - game['Date']).days
        weight = decay_factor ** days_ago
        point_diff = (game['P1 Total Points'] - game['P2 Total Points']) if game['Player 1 ID'] == p1_id else (game['P2 Total Points'] - game['P1 Total Points'])
        total_weighted_score += (point_diff * weight)
    return total_weighted_score


def calculate_pdr(player_id, rolling_games_df):
    if rolling_games_df.empty:
        return 0.5
    total_points_won, total_points_played = 0, 0
    for _, game in rolling_games_df.iterrows():
        if 'P1 Total Points' not in game or 'P2 Total Points' not in game:
            continue
        points_won = game['P1 Total Points'] if game['Player 1 ID'] == player_id else game['P2 Total Points']
        points_lost = game['P2 Total Points'] if game['Player 1 ID'] == player_id else game['P1 Total Points']
        total_points_won += points_won
        total_points_played += (points_won + points_lost)
    if total_points_played == 0:
        return 0.5
    return total_points_won / total_points_played


def calculate_performance_slope(performance_history):
    if len(performance_history) < 2:
        return 0.0
    y = np.array(performance_history)
    x = np.arange(len(y))
    return np.polyfit(x, y, 1)[0]


class TestBacktestHelperFunctions:
    """Tests for backtest helper functions."""

    def test_close_set_win_rate_empty(self):
        """Empty dataframe returns neutral 0.5."""
        result = calculate_close_set_win_rate(1, pd.DataFrame())
        assert result == 0.5

    def test_close_set_win_rate_all_won(self):
        """Player wins all close sets."""
        df = pd.DataFrame({
            'Player 1 ID': [1],
            'Player 2 ID': [2],
            'Set Scores': ['11-9,11-9'],
            'P1_Win': [1]
        })
        result = calculate_close_set_win_rate(1, df)
        assert result == 1.0

    def test_h2h_dominance_empty(self):
        """Empty H2H returns 0."""
        result = calculate_h2h_dominance(1, pd.DataFrame(), datetime.now(), 0.98)
        assert result == 0.0

    def test_h2h_dominance_positive(self):
        """Positive point differential gives positive score."""
        df = pd.DataFrame({
            'Player 1 ID': [1],
            'Player 2 ID': [2],
            'P1 Total Points': [60],
            'P2 Total Points': [40],
            'Date': [datetime.now() - timedelta(days=1)]
        })
        result = calculate_h2h_dominance(1, df, datetime.now(), 0.98)
        assert result > 0

    def test_pdr_empty(self):
        """Empty dataframe returns 0.5."""
        result = calculate_pdr(1, pd.DataFrame())
        assert result == 0.5

    def test_pdr_dominant(self):
        """Dominant player has PDR > 0.5."""
        df = pd.DataFrame({
            'Player 1 ID': [1],
            'Player 2 ID': [2],
            'P1 Total Points': [70],
            'P2 Total Points': [30]
        })
        result = calculate_pdr(1, df)
        assert result == 0.7

    def test_performance_slope_empty(self):
        """Empty or single value returns 0."""
        assert calculate_performance_slope([]) == 0.0
        assert calculate_performance_slope([0.5]) == 0.0

    def test_performance_slope_positive(self):
        """Increasing values have positive slope."""
        result = calculate_performance_slope([0.4, 0.5, 0.6])
        assert result > 0

    def test_performance_slope_negative(self):
        """Decreasing values have negative slope."""
        result = calculate_performance_slope([0.6, 0.5, 0.4])
        assert result < 0


class TestKellyBetting:
    """Tests for Kelly criterion calculations."""

    def test_kelly_positive_edge(self):
        """Positive edge should recommend a stake."""
        # Kelly formula: f = (p * b - q) / b
        # where p = win prob, q = 1-p, b = odds - 1
        p = 0.55  # 55% win probability
        odds = 2.0  # Even money
        b = odds - 1
        q = 1 - p
        kelly = (p * b - q) / b
        assert kelly > 0

    def test_kelly_no_edge(self):
        """No edge should recommend no stake."""
        p = 0.5
        odds = 2.0
        b = odds - 1
        q = 1 - p
        kelly = (p * b - q) / b
        assert abs(kelly) < 0.01  # Near zero

    def test_kelly_negative_edge(self):
        """Negative edge should recommend no stake."""
        p = 0.45
        odds = 2.0
        b = odds - 1
        q = 1 - p
        kelly = (p * b - q) / b
        assert kelly < 0


class TestVectorizedEquivalence:
    """Tests to verify vectorized versions match original implementations."""

    def test_pdr_vectorized_matches_original(self):
        """Vectorized PDR should match loop-based PDR."""
        df = pd.DataFrame({
            'Player 1 ID': [1, 1, 2],
            'Player 2 ID': [2, 3, 1],
            'P1 Total Points': [50, 60, 45],
            'P2 Total Points': [45, 55, 55]
        })

        # Original loop-based
        original_result = calculate_pdr(1, df)

        # Vectorized approach
        is_p1 = df['Player 1 ID'] == 1
        points_won = np.where(is_p1, df['P1 Total Points'], df['P2 Total Points'])
        points_lost = np.where(is_p1, df['P2 Total Points'], df['P1 Total Points'])
        vectorized_result = points_won.sum() / (points_won.sum() + points_lost.sum())

        assert abs(original_result - vectorized_result) < 0.001

    def test_h2h_wins_vectorized_matches_original(self):
        """Vectorized H2H wins calculation should match apply-based."""
        df = pd.DataFrame({
            'Player 1 ID': [1, 2, 1],
            'Player 2 ID': [2, 1, 2],
            'P1_Win': [1, 0, 1]  # P1 wins when they're P1, P2 wins when they're P1
        })
        p1_id = 1

        # Original apply-based
        original = df.apply(
            lambda r: 1 if (r['Player 1 ID'] == p1_id and r['P1_Win'] == 1) or
                          (r['Player 2 ID'] == p1_id and r['P1_Win'] == 0) else 0,
            axis=1
        ).sum()

        # Vectorized
        p1_was_player1 = df['Player 1 ID'].values == p1_id
        p1_won_as_player1 = p1_was_player1 & (df['P1_Win'].values == 1)
        p1_won_as_player2 = (~p1_was_player1) & (df['P1_Win'].values == 0)
        vectorized = np.sum(p1_won_as_player1 | p1_won_as_player2)

        assert original == vectorized


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
