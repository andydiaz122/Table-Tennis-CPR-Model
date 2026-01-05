CPR Model v7.3 - Final Project Archive
This archive contains all the necessary Python scripts for the Czech Liga Pro table tennis predictive model, version 7.3. The project has undergone a complete debugging and validation cycle to ensure that the back-testing methodology is sound and perfectly synchronized with the live prediction script.

The Workflow
The project follows a logical progression from data collection to final prediction.

Data Collection (betsapi_ligapro_collector_FIXED.py)

This script fetches raw match data from the BetsAPI for the Czech Liga Pro.

It has been corrected to handle unexpected data formats (e.g., non-integer scores) from the API.

It intelligently resumes downloads and avoids duplicating matches that are already in the output CSV.

Data Analysis & Strategy Discovery

Log Generation (backtest_for_analysis_v7.3.py): This is a critical script. It runs a point-in-time correct back-test with very wide filters. Its purpose is not to simulate a real strategy, but to generate a large, accurate log file (backtest_log_for_analysis_v7.3.csv) of the model's raw performance across many different scenarios. It was crucial in fixing logging errors and the "runaway compounding" bug.

Performance Analysis (analyze_performance_FIXED.py): This script reads the log file generated above and creates a detailed forensic report, breaking down the model's ROI by various factors like odds, perceived edge, and H2H advantage. This report is what we used to discover the final, profitable v7.3 strategy.

Final Strategy Validation (backtest_final_v7.3.py)

This script implements the final, intelligent v7.3 filters discovered from the analysis (Edge > 10% and H2H Win Rate >= 40%).

It runs a full, point-in-time correct simulation of this exact strategy, providing a reliable measure of its historical performance (ROI, profit, etc.).

It also outputs a log (backtest_log_final_v7.3.csv) of only the bets that this final strategy would have placed.

Live Prediction (final_predictor_v7.3.py)

This is the script to use for making daily predictions.

It is perfectly synchronized with backtest_final_v7.3.py, using the identical models, feature calculations, and strategic filters.

You manually input the upcoming matches and their odds, and the script provides a clear "BET" or "NO BET" recommendation based on the validated v7.3 strategy.