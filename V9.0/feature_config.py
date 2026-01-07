# Feature Toggle Configuration for CPR Model V9.0
# Systematic Feature Removal Analysis
#
# Usage: Toggle features True (include) or False (exclude) for testing
# Each test should modify ONE feature at a time, then run the pipeline

FEATURE_TOGGLES = {
    # === FEATURES TO REMOVE (6) - Zero or positive impact when removed ===
    'Time_Since_Last_Advantage': False,  # REMOVE - ROI 2.44% (0% impact)
    'Matches_Last_24H_Advantage': False,  # REMOVE - ROI 2.44% (0% impact)
    'Is_First_Match_Advantage': False,  # REMOVE - ROI 2.44% (0% impact)
    'PDR_Slope_Advantage': False,  # REMOVE - ROI 2.44% (0% impact)
    'Daily_Fatigue_Advantage': False,  # REMOVE - ROI 2.43% (-0.01%)
    'Win_Rate_Advantage': False,  # REMOVE - ROI 2.47% (+0.03% improvement!)

    # === FEATURES TO KEEP (6) - Significant negative impact when removed ===
    'H2H_P1_Win_Rate': True,  # KEEP - ROI 1.88% (-0.56%) CRITICAL!
    'H2H_Dominance_Score': True,  # KEEP - ROI 2.31% (-0.13%)
    'PDR_Advantage': True,  # KEEP - ROI 2.35% (-0.09%)
    'Win_Rate_L5_Advantage': True,  # KEEP - ROI 2.31% (-0.13%)
    'Close_Set_Win_Rate_Advantage': True,  # KEEP - ROI 2.34% (-0.10%)
    'Set_Comebacks_Advantage': True,  # KEEP - ROI 2.43% (-0.01%)
}


def get_active_features():
    """Returns list of features that are toggled ON"""
    return [f for f, enabled in FEATURE_TOGGLES.items() if enabled]


def get_disabled_features():
    """Returns list of features that are toggled OFF"""
    return [f for f, enabled in FEATURE_TOGGLES.items() if not enabled]


def get_experiment_name():
    """Returns a descriptive name for current configuration"""
    disabled = get_disabled_features()
    if not disabled:
        return "Baseline_12_Features"
    return f"Without_{'_'.join(disabled)}"


def get_feature_count():
    """Returns count of active features"""
    return len(get_active_features())


# Print configuration when module is imported
if __name__ == "__main__":
    print(f"Experiment: {get_experiment_name()}")
    print(f"Active features ({get_feature_count()}):")
    for f in get_active_features():
        print(f"  - {f}")

    disabled = get_disabled_features()
    if disabled:
        print(f"\nDisabled features ({len(disabled)}):")
        for f in disabled:
            print(f"  - {f}")
