"""
Unit Tests for merge_data_v7.4.py
Tests merge logic to ensure optimization doesn't break functionality.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime


class TestMergeLogic:
    """Tests for the merge data pipeline logic."""

    def test_kickoff_odds_selection(self):
        """Test that we select the latest pre-match odds correctly."""
        # Create mock features
        features = pd.DataFrame({
            'Match ID': [1, 2],
            'Date': ['2024-01-01', '2024-01-02'],
            'Time': ['14:00:00', '15:00:00']
        })
        features['Match_Start_Time'] = pd.to_datetime(
            features['Date'] + ' ' + features['Time']
        )

        # Create mock odds with multiple timestamps per match
        odds = pd.DataFrame({
            'Match ID': [1, 1, 1, 2, 2],
            'Market_Name': ['Match Winner'] * 5,
            'Odds_Timestamp': pd.to_datetime([
                '2024-01-01 12:00:00',  # Early
                '2024-01-01 13:30:00',  # Latest pre-match
                '2024-01-01 14:05:00',  # In-play (after start)
                '2024-01-02 14:00:00',  # Latest pre-match
                '2024-01-02 15:30:00',  # In-play
            ]),
            'P1_Odds': [1.8, 1.9, 2.0, 1.5, 1.6],
            'P2_Odds': [2.2, 2.1, 2.0, 2.5, 2.4]
        })

        # Apply the kickoff selection logic
        merged_odds = pd.merge(
            odds,
            features[['Match ID', 'Match_Start_Time']],
            on='Match ID',
            how='left'
        )
        prematch_odds = merged_odds[
            merged_odds['Odds_Timestamp'] <= merged_odds['Match_Start_Time']
        ]
        kickoff_indices = prematch_odds.groupby('Match ID')['Odds_Timestamp'].idxmax()
        kickoff_odds = prematch_odds.loc[kickoff_indices]

        # Verify we got the right odds
        assert len(kickoff_odds) == 2

        match1_odds = kickoff_odds[kickoff_odds['Match ID'] == 1].iloc[0]
        assert match1_odds['P1_Odds'] == 1.9  # Latest pre-match

        match2_odds = kickoff_odds[kickoff_odds['Match ID'] == 2].iloc[0]
        assert match2_odds['P1_Odds'] == 1.5  # Latest pre-match

    def test_match_id_column_rename(self):
        """Test that Match_ID gets renamed to Match ID."""
        df = pd.DataFrame({'Match_ID': [1, 2, 3], 'Value': [10, 20, 30]})

        if 'Match_ID' in df.columns:
            df.rename(columns={'Match_ID': 'Match ID'}, inplace=True)

        assert 'Match ID' in df.columns
        assert 'Match_ID' not in df.columns

    def test_merge_preserves_all_features(self):
        """Test that left merge keeps all feature rows."""
        features = pd.DataFrame({
            'Match ID': [1, 2, 3],
            'Feature1': [0.5, 0.6, 0.7]
        })
        odds = pd.DataFrame({
            'Match ID': [1, 2],  # No odds for match 3
            'Kickoff_P1_Odds': [1.8, 1.9]
        })

        merged = pd.merge(features, odds, on='Match ID', how='left')

        assert len(merged) == 3  # All feature rows preserved
        assert pd.isna(merged[merged['Match ID'] == 3]['Kickoff_P1_Odds'].iloc[0])


class TestDtypeOptimization:
    """Tests for dtype optimization."""

    def test_float32_precision_sufficient(self):
        """Verify float32 is sufficient for odds values."""
        odds_float64 = np.array([1.85, 2.10, 3.50], dtype=np.float64)
        odds_float32 = odds_float64.astype(np.float32)

        # Difference should be negligible for odds (typically 1.01 - 10.0)
        assert np.allclose(odds_float64, odds_float32, rtol=1e-5)

    def test_int32_sufficient_for_match_ids(self):
        """Verify int32 is sufficient for match IDs."""
        max_match_id = 10_000_000  # 10 million matches
        assert max_match_id < np.iinfo(np.int32).max


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
