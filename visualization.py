import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load processed data
df = pd.read_csv('X_train.csv')

# Create unique admissions dataframe for visualization
unique_admissions = df.drop_duplicates(subset=['ICUSTAY_ID'])

# Create histogram of age distribution for unique admissions
plt.figure(figsize=(10, 6))
sns.histplot(data=unique_admissions, x='AGE', kde=True, bins=20)
plt.title('Age Distribution of Unique Admissions')
plt.xlabel('Age (years)')
plt.ylabel('Number of Admissions')
plt.tight_layout()
plt.close()

# Create gender distribution plot
plt.figure(figsize=(10, 6))
sns.countplot(x='GENDER', data=unique_admissions, palette="Set2", hue='GENDER')
plt.title('Gender Distribution of Unique Admissions')
plt.xlabel('Gender')
plt.ylabel('Number of Admissions')
plt.close()

# Create mortality distribution plot
plt.figure(figsize=(10, 6))
mortality_counts = unique_admissions['IN_HOSPITAL_MORTALITY'].value_counts(normalize=True) * 100
sns.countplot(x='IN_HOSPITAL_MORTALITY', data=unique_admissions, palette='Set2', 
              hue='IN_HOSPITAL_MORTALITY', legend=False)
plt.title('In-Hospital Mortality Rate (Unique Admissions)')
plt.xlabel('In-Hospital Mortality')
plt.ylabel('Number of Admissions')
labels = [f'Survived ({mortality_counts.get(0,0):.1f}%)',
          f'Expired ({mortality_counts.get(1,0):.1f}%)']
plt.xticks([0, 1], labels)
plt.close()

# Create admission type distribution plot
plt.figure(figsize=(10, 6))
sns.countplot(y='ADMISSION_TYPE', data=unique_admissions, 
              order=unique_admissions['ADMISSION_TYPE'].value_counts().index, 
              palette="Set2", hue="ADMISSION_TYPE")
plt.title('Admission Type Distribution (Unique Admissions)')
plt.xlabel('Number of Admissions')
plt.ylabel('Admission Type')
plt.tight_layout()
plt.close()

print("Visualizations complete. Saved to PNG files.")
