import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set paths to data
data_path = '/content/drive/My Drive/mimic_iii_data_raw/'
admission_path = data_path + 'ADMISSIONS.csv.gz'
patients_path = data_path + 'PATIENTS.csv.gz'
icu_stays_path = data_path + 'ICUSTAYS.csv.gz'
note_events_path= data_path + 'NOTEEVENTS.csv.gz'
chart_events_path = data_path + 'CHARTEVENTS.csv.gz'

# Mount Google Drive (if using Colab)
try:
    from google.colab import drive
    drive.mount('/content/drive')
except:
    print("Not running in Colab or drive already mounted")

# Load patients data
patients_df = pd.read_csv(patients_path, compression='gzip')
patients_df['DOB'] = pd.to_datetime(patients_df['DOB'])
patients_df = patients_df[['SUBJECT_ID', 'GENDER', 'DOB']]

# Load admissions data
admissions_df = pd.read_csv(admission_path, compression='gzip')
admissions_df['ADMITTIME'] = pd.to_datetime(admissions_df['ADMITTIME'])
admissions_df['DISCHTIME'] = pd.to_datetime(admissions_df['DISCHTIME'])
admissions_df['DEATHTIME'] = pd.to_datetime(admissions_df['DEATHTIME'])

# Merge admissions and patients data
df = pd.merge(admissions_df, patients_df, on='SUBJECT_ID', how='inner')

# Free memory
del admissions_df
del patients_df
gc.collect()

# Calculate age and filter adult patients
df['AGE'] = (df['ADMITTIME'].dt.year - df['DOB'].dt.year)
df = df[df['AGE'] >= 18]
df = df.sort_values(by=['SUBJECT_ID', 'ADMITTIME'])

# Keep first admission per patient with chart events
df = df.drop_duplicates(subset=['SUBJECT_ID'], keep='first')
df = df[df['HAS_CHARTEVENTS_DATA'] == 1]
df = df.sample(n=2000, random_state=42)

# Select relevant columns and create target variable
df['IN_HOSPITAL_MORTALITY'] = df['HOSPITAL_EXPIRE_FLAG']
df = df[['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME', 
         'ADMISSION_TYPE', 'DIAGNOSIS', 'AGE', 'GENDER', 'IN_HOSPITAL_MORTALITY']]

# Load ICU data
icustays_df = pd.read_csv(icu_stays_path, compression='gzip')
icustays_df['INTIME'] = pd.to_datetime(icustays_df['INTIME'])
icustays_df['OUTTIME'] = pd.to_datetime(icustays_df['OUTTIME'])

# Clean ICU data
cohort_icustays = pd.merge(icustays_df, df[['HADM_ID']], on=['HADM_ID'], how='inner')
del icustays_df
gc.collect()
cohort_icustays = cohort_icustays.sort_values(by=['HADM_ID', 'INTIME'])

# Get first ICU stay per admission
first_icustays = cohort_icustays.drop_duplicates(subset=['HADM_ID'], keep='first')
first_icustays = first_icustays[['HADM_ID', 'ICUSTAY_ID', 'INTIME', 'OUTTIME', 'LOS', 'FIRST_CAREUNIT']]

# Merge ICU data with main dataframe
df = pd.merge(df, first_icustays, on=['HADM_ID'], how='inner')
del first_icustays
gc.collect()

# Function to load chart events for the cohort
def load_chartevents_for_cohort(chart_events_path, cohort_icustay_ids, max_chunks=1, chunk_size=5000000):
    cohort_ids = set(cohort_icustay_ids)
    filtered_chartevents = []

    columns_to_read = ['ICUSTAY_ID', 'ITEMID', 'CHARTTIME', 'VALUE', 'VALUENUM', 'VALUEUOM']

    chartevents_reader = pd.read_csv(
        chart_events_path,
        compression='gzip',
        usecols=columns_to_read,
        chunksize=chunk_size,
        nrows=chunk_size * max_chunks
    )

    for chunk_idx, chunk in enumerate(chartevents_reader, 1):
        start_row = (chunk_idx - 1) * chunk_size + 1
        end_row = chunk_idx * chunk_size
        print(f"  Processing chartevents chunk {chunk_idx} (rows {start_row} to {end_row})...")

        relevant_records = chunk[chunk['ICUSTAY_ID'].isin(cohort_ids)]

        if not relevant_records.empty:
            filtered_chartevents.append(relevant_records)

        if chunk_idx >= max_chunks:
            print(f"  Stopped reading CHARTEVENTS after {max_chunks} chunk(s).")
            break

    del chartevents_reader
    gc.collect()

    return filtered_chartevents

# Load chart events for the cohort
cohort_icustay_ids = df['ICUSTAY_ID'].unique()
raw_chartevents_for_cohort = load_chartevents_for_cohort(chart_events_path, cohort_icustay_ids)

all_raw_events_df = pd.concat(raw_chartevents_for_cohort, ignore_index=True)
all_raw_events_df["ICUSTAY_ID"] = all_raw_events_df["ICUSTAY_ID"].astype(int)

# Merge chart events with main dataframe
df = pd.merge(df, all_raw_events_df, on='ICUSTAY_ID', how='inner')
del all_raw_events_df
del raw_chartevents_for_cohort
gc.collect()

# Function to load note events for the cohort
def load_noteevents_for_cohort(note_events_path, cohort_hadm_ids, max_chunks=1, chunk_size=500000):
    cohort_ids = set(cohort_hadm_ids)
    filtered_noteevents = []

    columns_to_read = [
        'ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'CHARTDATE', 'CHARTTIME',
        'STORETIME', 'CATEGORY', 'DESCRIPTION', 'CGID', 'ISERROR', 'TEXT'
    ]

    noteevents_reader = pd.read_csv(
        note_events_path,
        compression='gzip',
        usecols=columns_to_read,
        chunksize=chunk_size,
        nrows=chunk_size * max_chunks
    )

    for chunk_idx, chunk in enumerate(noteevents_reader, 1):
        start_row = (chunk_idx - 1) * chunk_size + 1
        end_row = chunk_idx * chunk_size
        print(f"  Processing noteevents chunk {chunk_idx} (rows {start_row} to {end_row})...")

        relevant_notes = chunk[chunk['HADM_ID'].isin(cohort_ids)]

        if not relevant_notes.empty:
            filtered_noteevents.append(relevant_notes)

        if chunk_idx >= max_chunks:
            print(f"  Stopped reading NOTEEVENTS after {max_chunks} chunk(s).")
            break

    del noteevents_reader
    gc.collect()

    return filtered_noteevents

# Load note events for the cohort
cohort_hadm_ids = df['HADM_ID'].unique()
raw_noteevents_for_cohort = load_noteevents_for_cohort(note_events_path, cohort_hadm_ids)

all_raw_notes_df = pd.concat(raw_noteevents_for_cohort, ignore_index=True)
all_raw_notes_df["HADM_ID"] = all_raw_notes_df["HADM_ID"].astype(int)
all_raw_notes_df = all_raw_notes_df[["HADM_ID", "CHARTDATE", "CATEGORY", "DESCRIPTION", "TEXT"]]

# Combine notes by admission
notes_combined = all_raw_notes_df.groupby('HADM_ID').agg({
    'TEXT': lambda x: ' '.join(str(text) for text in x if pd.notna(text))
}).reset_index()

# Merge notes with main dataframe
df_merged = pd.merge(df, notes_combined, on='HADM_ID', how='inner')

# Create final dataset with aggregated values
df_final = df_merged.groupby('ICUSTAY_ID').agg({
    'VALUENUM': 'mean',
    'TEXT': 'first',
    'IN_HOSPITAL_MORTALITY': 'first'
}).reset_index()
df_final = df_final.dropna(subset=['VALUENUM', 'TEXT', 'IN_HOSPITAL_MORTALITY'])

# Split into train and test sets
X_train, X_test = train_test_split(df_final, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train['VALUENUM'] = scaler.fit_transform(X_train[['VALUENUM']])
X_test['VALUENUM'] = scaler.transform(X_test[['VALUENUM']])

# Save processed data
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)

print("Data processing complete. Saved X_train.csv and X_test.csv")
