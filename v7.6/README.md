CPR Model v7.4 - Final Archive
This archive contains the complete set of scripts for the CPR Model v7.4. This version is the result of a rigorous debugging and validation process, ensuring that the model is symmetrical, the back-testing is point-in-time correct, and the live predictions are perfectly synchronized with the proven strategy.

The Workflow
The scripts are designed to be used in a specific order to go from raw data to actionable predictions.

Phase 1: Strategy Discovery & Validation (Periodic Task)

This phase is performed periodically (e.g., monthly) or whenever you want to re-evaluate your strategy based on new data.

backtest_for_analysis_v7.4.py:

Purpose: To generate a comprehensive performance log with wide filters.

Action: Run this script on your final_dataset_v7.1.csv. It will produce backtest_log_for_analysis_v7.4.csv. Its goal is to capture as much data as possible about the model's raw performance.

analyze_performance_FIXED.py:

Purpose: To analyze the raw performance log and discover the model's strengths and weaknesses.

Action: Run this script after generating the analysis log. It will read backtest_log_for_analysis_v7.4.csv and print a detailed forensic report to the console. Use this report to determine the optimal strategic filters (e.g., EDGE_THRESHOLD_MIN, H2H_DISADVANTAGE_THRESHOLD).

Phase 2: Final Strategy Simulation (One-Time Verification)

This phase confirms the performance of your chosen filters.

backtest_final_v7.4.py:

Purpose: To run a definitive back-test of the final, optimized strategy.

Action: Update the filter constants in this script based on your findings from the analysis. Run it to get the final, expected ROI and performance metrics of your live strategy.

Phase 3: Daily Live Prediction

This is your daily operational workflow.

betsapi_ligapro_collector_FIXED.py:

Purpose: To collect the latest match results.

Action: Run this daily to keep your czech_liga_pro_advanced_stats_FIXED.csv up to date.

(Not Included) advanced_feature_engineering.py & split_data.py:

Purpose: To process the new raw data into features for the model.

Action: After collecting new data, run your feature engineering pipeline to update final_dataset_v7.1.csv.

final_predictor_v7.4.py:

Purpose: To generate predictions for upcoming matches using the final, optimized v7.4 strategy.

Action: This is your live prediction script. Manually update the upcoming_matches list with the day's games and run it. It is perfectly synchronized with backtest_final_v7.4.py and will execute your proven strategy.

This structured process ensures that your live predictions are always based on a rigorously tested and validated strategy.