import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from picai_baseline.splits.picai import train_splits, valid_splits

# Load clinical data
df = pd.read_csv('/HOME/alshaart/ITUNet-for-PICAI-2022-Challenge/picai_labels/clinical_information/marksheet.csv')

print("Data types:")
print(df.dtypes)
print(f"\nTotal patients in dataset: {len(df)}")

# Create fold assignments from official PI-CAI splits
fold_assignments = {}
for fold in train_splits.keys():
    all_subjects_in_fold = train_splits[fold]['subject_list'] + valid_splits[fold]['subject_list']
    fold_assignments[f'fold{fold}'] = [int(subject) for subject in all_subjects_in_fold]

print(f"\nFold assignments from PI-CAI splits:")
for fold_name, patient_ids in fold_assignments.items():
    print(f"{fold_name}: {len(patient_ids)} patients")

# Add fold information to dataframe
df['fold'] = None
patients_assigned = 0
for fold_name, patient_ids in fold_assignments.items():
    mask = df['patient_id'].isin(patient_ids)
    df.loc[mask, 'fold'] = fold_name
    patients_assigned += mask.sum()

print(f"\nPatients assigned to folds: {patients_assigned}")
print(f"Patients without fold assignment: {df['fold'].isna().sum()}")

# Check if any patients got assigned
if patients_assigned == 0:
    print("\n⚠ WARNING: No patients were assigned to folds!")
    print("This might be because patient IDs in the splits don't match the CSV file.")
    print("Let's check the first few patient IDs in both datasets:")
    print(f"First 10 patient IDs in CSV: {df['patient_id'].head(10).tolist()}")
    print(f"First 10 patient IDs in fold0: {fold_assignments['fold0'][:10]}")
    
    # Try alternative approach - check if we need to match study_id instead
    print(f"\nFirst 10 study IDs in CSV: {df['study_id'].head(10).tolist()}")
    
    # Try matching with study_id instead
    print("\nTrying to match with study_id...")
    df['fold'] = None
    patients_assigned = 0
    for fold_name, patient_ids in fold_assignments.items():
        mask = df['study_id'].isin(patient_ids)
        df.loc[mask, 'fold'] = fold_name
        patients_assigned += mask.sum()
    
    print(f"Patients assigned to folds using study_id: {patients_assigned}")

# Function to determine cancer status from lesion_ISUP
def get_cancer_status(lesion_isup):
    if pd.isna(lesion_isup):
        return 'No Cancer'
    
    # Convert to string and handle comma-separated values
    lesion_str = str(lesion_isup)
    
    # Split by comma and convert to numbers, ignoring non-numeric values
    lesions = []
    for item in lesion_str.split(','):
        try:
            num = float(item.strip())
            lesions.append(num)
        except (ValueError, TypeError):
            continue
    
    # If any lesion has ISUP > 0, patient has cancer
    if any(lesion > 0 for lesion in lesions):
        return 'Cancer'
    else:
        return 'No Cancer'

# Function to get highest ISUP grade (for cancer severity)
def get_highest_isup(lesion_isup):
    if pd.isna(lesion_isup):
        return 0
    
    lesion_str = str(lesion_isup)
    lesions = []
    for item in lesion_str.split(','):
        try:
            num = float(item.strip())
            lesions.append(num)
        except (ValueError, TypeError):
            continue
    
    return max(lesions) if lesions else 0

# Apply the functions
df['cancer_status'] = df['lesion_ISUP'].apply(get_cancer_status)
df['highest_isup'] = df['lesion_ISUP'].apply(get_highest_isup)

print(f"\nCancer status distribution:")
print(df['cancer_status'].value_counts())
print(f"\nHighest ISUP distribution:")
print(df['highest_isup'].value_counts().sort_index())

# Check if we have any patients with fold assignments
if df['fold'].notna().sum() == 0:
    print("\n❌ ERROR: Still no patients assigned to folds.")
    print("Creating a mock fold assignment for demonstration...")
    # Create mock fold assignments for demonstration
    np.random.seed(42)
    folds = ['fold0', 'fold1', 'fold2', 'fold3', 'fold4']
    df['fold'] = np.random.choice(folds, size=len(df))
    print("Using mock fold assignments for visualization.")

# Create comprehensive cancer distribution visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Cancer vs No-Cancer Distribution Across PI-CAI Folds', fontsize=16, fontweight='bold')

# 1. Cancer vs No-Cancer Count per Fold (Stacked Bar)
cancer_cross_tab = pd.crosstab(df['fold'], df['cancer_status'])
print(f"\nCancer cross-tabulation:")
print(cancer_cross_tab)

if cancer_cross_tab.empty:
    print("❌ Cross-tabulation is empty. Cannot create plot.")
else:
    colors = ['lightcoral', 'lightgreen']  # Cancer, No Cancer
    cancer_cross_tab.plot(kind='bar', stacked=True, ax=axes[0,0], color=colors)
    axes[0,0].set_title('Cancer vs No-Cancer Distribution\n(Counts per Fold)', fontweight='bold')
    axes[0,0].set_xlabel('Fold')
    axes[0,0].set_ylabel('Number of Patients')
    axes[0,0].legend(title='Cancer Status', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0,0].grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (fold, row) in enumerate(cancer_cross_tab.iterrows()):
        cancer_count = row['Cancer']
        no_cancer_count = row['No Cancer']
        axes[0,0].text(i, cancer_count/2, f'{cancer_count}', ha='center', va='center', fontweight='bold', color='white')
        axes[0,0].text(i, cancer_count + no_cancer_count/2, f'{no_cancer_count}', ha='center', va='center', fontweight='bold', color='darkgreen')

    # 2. Cancer vs No-Cancer Percentage per Fold
    cancer_cross_tab_pct = pd.crosstab(df['fold'], df['cancer_status'], normalize='index') * 100
    cancer_cross_tab_pct.plot(kind='bar', stacked=True, ax=axes[0,1], color=colors)
    axes[0,1].set_title('Cancer vs No-Cancer Distribution\n(Percentage per Fold)', fontweight='bold')
    axes[0,1].set_xlabel('Fold')
    axes[0,1].set_ylabel('Percentage (%)')
    axes[0,1].legend(title='Cancer Status', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0,1].grid(axis='y', alpha=0.3)

    # Add percentage labels on bars
    for i, (fold, row) in enumerate(cancer_cross_tab_pct.iterrows()):
        cancer_pct = row['Cancer']
        no_cancer_pct = row['No Cancer']
        axes[0,1].text(i, cancer_pct/2, f'{cancer_pct:.1f}%', ha='center', va='center', fontweight='bold', color='white')
        axes[0,1].text(i, cancer_pct + no_cancer_pct/2, f'{no_cancer_pct:.1f}%', ha='center', va='center', fontweight='bold', color='darkgreen')

    # 3. Cancer Rate Comparison Across Folds
    cancer_rates = cancer_cross_tab_pct['Cancer']
    bars = axes[1,0].bar(cancer_rates.index, cancer_rates.values, color='lightcoral', alpha=0.7)
    axes[1,0].set_title('Cancer Rate Comparison Across Folds', fontweight='bold')
    axes[1,0].set_xlabel('Fold')
    axes[1,0].set_ylabel('Cancer Rate (%)')
    axes[1,0].grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        axes[1,0].text(bar.get_x() + bar.get_width()/2., height + 1,
                      f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

    # Add average line
    avg_cancer_rate = cancer_rates.mean()
    axes[1,0].axhline(y=avg_cancer_rate, color='red', linestyle='--', alpha=0.7, 
                      label=f'Average: {avg_cancer_rate:.1f}%')
    axes[1,0].legend()

    # 4. Detailed Cancer Severity Breakdown (ISUP Grades 1-5)
    cancer_cases = df[df['cancer_status'] == 'Cancer']
    # Use highest_isup for cancer severity
    isup_cross_tab = pd.crosstab(cancer_cases['fold'], cancer_cases['highest_isup'])
    isup_colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(isup_cross_tab.columns)))

    isup_cross_tab.plot(kind='bar', stacked=True, ax=axes[1,1], color=isup_colors)
    axes[1,1].set_title('Cancer Severity Breakdown (Highest ISUP 1-5)\nper Fold', fontweight='bold')
    axes[1,1].set_xlabel('Fold')
    axes[1,1].set_ylabel('Number of Cancer Cases')
    axes[1,1].legend(title='Highest ISUP', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1,1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    # Save the plot as an image file
    plt.savefig('/HOME/alshaart/ITUNet-for-PICAI-2022-Challenge/cancer_distribution_across_folds.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('/HOME/alshaart/ITUNet-for-PICAI-2022-Challenge/cancer_distribution_across_folds.pdf', 
                bbox_inches='tight', facecolor='white')

    print("Plot saved as:")
    print("- /HOME/alshaart/ITUNet-for-PICAI-2022-Challenge/cancer_distribution_across_folds.png")
    print("- /HOME/alshaart/ITUNet-for-PICAI-2022-Challenge/cancer_distribution_across_folds.pdf")

    # Close the plot to free memory
    plt.close()

# Print detailed statistics
print("="*70)
print("CANCER vs NO-CANCER DISTRIBUTION ACROSS FOLDS")
print("="*70)

for fold in sorted(df['fold'].unique()):
    fold_data = df[df['fold'] == fold]
    total_patients = len(fold_data)
    
    cancer_cases = fold_data[fold_data['cancer_status'] == 'Cancer']
    no_cancer_cases = fold_data[fold_data['cancer_status'] == 'No Cancer']
    
    cancer_count = len(cancer_cases)
    no_cancer_count = len(no_cancer_cases)
    
    if total_patients > 0:
        cancer_rate = (cancer_count / total_patients) * 100
        no_cancer_rate = (no_cancer_count / total_patients) * 100
    else:
        cancer_rate = no_cancer_rate = 0
    
    print(f"\n{fold.upper()}:")
    print(f"  Total patients: {total_patients}")
    print(f"  Cancer cases: {cancer_count} ({cancer_rate:.1f}%)")
    print(f"  No-cancer cases: {no_cancer_count} ({no_cancer_rate:.1f}%)")
    
    # Cancer severity breakdown
    if cancer_count > 0:
        print(f"  Cancer severity (Highest ISUP grades):")
        for isup_grade in sorted(cancer_cases['highest_isup'].unique()):
            if isup_grade > 0:  # Only show cancer grades
                grade_count = (cancer_cases['highest_isup'] == isup_grade).sum()
                grade_percentage = (grade_count / cancer_count) * 100
                print(f"    ISUP {int(isup_grade)}: {grade_count} cases ({grade_percentage:.1f}%)")

# Overall statistics
print("\n" + "="*70)
print("OVERALL CANCER STATISTICS")
print("="*70)
total_patients = len(df)
total_cancer = len(df[df['cancer_status'] == 'Cancer'])
total_no_cancer = len(df[df['cancer_status'] == 'No Cancer'])

print(f"Total patients: {total_patients}")
print(f"Total cancer cases: {total_cancer} ({(total_cancer/total_patients)*100:.1f}%)")
print(f"Total no-cancer cases: {total_no_cancer} ({(total_no_cancer/total_patients)*100:.1f}%)")

# Balance analysis
print("\nFOLD BALANCE ANALYSIS (Cancer Distribution):")
cancer_rates = []
for fold in sorted(df['fold'].unique()):
    fold_data = df[df['fold'] == fold]
    if len(fold_data) > 0:
        cancer_rate = (fold_data['cancer_status'] == 'Cancer').mean() * 100
        cancer_rates.append(cancer_rate)
        print(f"  {fold}: {cancer_rate:.1f}% cancer rate")

if cancer_rates:
    avg_cancer_rate = np.mean(cancer_rates)
    std_cancer_rate = np.std(cancer_rates)
    print(f"\nAverage cancer rate: {avg_cancer_rate:.1f}%")
    print(f"Standard deviation: {std_cancer_rate:.1f}%")
    print(f"Max-Min difference: {max(cancer_rates)-min(cancer_rates):.1f}%")

    if std_cancer_rate < 5:
        print("✓ Good balance: Cancer rates are well distributed across folds")
    else:
        print("⚠ Moderate imbalance: Consider checking fold stratification")

