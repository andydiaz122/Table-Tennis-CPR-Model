import pandas as pd

# Load the dataset
file_path = 'final_dataset_v7.4.csv'
df = pd.read_csv(file_path)

print(f"Dataset shape before removing duplicates: {df.shape}")

# ⚠️ STEP 1: Define the columns that uniquely identify an observation.
# Replace 'Col_A', 'Col_B', 'Col_C' with the actual names of your key columns.
# key_columns = ['Match ID', 'Date', 'Time', 'Player 1 ID', 'Player 2 ID']

# ⚠️ STEP 2: Remove duplicates based ONLY on the values in the key_columns.
# 'keep=False' will remove ALL duplicates, but 'keep=first' (default) is safer.
df.drop_duplicates(subset='Match ID', keep='last', inplace=True)

print(f"Dataset shape after removing duplicates: {df.shape}")

# Save the cleaned data
df.to_csv('final_dataset_v7.4_no_duplicates.csv', index=False)

print(f"Duplicates removed (based on {'Match ID'}) and data saved.")


# 10745707	9/28/2025	10:30:00	1015755	Jiri Koch	768461	Oskar Spacek	2-Mar	11-8, 13-15, 8-11, 12-10, 11-4	5	55	48	4	2	0	1	1	0.7	0.7	0.4	1.3	0	0	1
# 10745707	9/28/2025	10:30:00	1015755	Jiri Koch	768461	Oskar Spacek	2-Mar	11-8, 13-15, 8-11, 12-10, 11-4	5	55	48	4	2	0	1	1	0.7	0.8	0.4	1.3	0	0	1
# 10745707	9/28/2025	10:30:00	1015755	Jiri Koch	768461	Oskar Spacek	2-Mar	11-8, 13-15, 8-11, 12-10, 11-4	5	55	48	4	2	0	1	1	0.7	0.8	0.4	1.3	0	0	1	2	1.727
# 10745707	9/28/2025	10:30:00	1015755	Jiri Koch	768461	Oskar Spacek	3-2	11-8, 13-15, 8-11, 12-10, 11-4	5	55	48	4	2	0	1	1	0.7	0.8	0.4	1.3	0	0	1	2	1.727
