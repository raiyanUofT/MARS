import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def collect_roc_data_per_exercise(results_dir, model_name, class_labels):
    """
    Collects ROC data per exercise (class label) from all subjects and saves them into separate CSV files.

    Parameters:
    - results_dir (str): The directory where the results are stored.
    - model_name (str): The model name whose ROC data to process.
    - class_labels (list): List of class labels (exercise names).
    """
    # Create a directory to store the per-exercise ROC data
    per_exercise_dir = os.path.join(results_dir, f'{model_name.lower().replace(" ", "_")}_per_exercise_roc_data')
    os.makedirs(per_exercise_dir, exist_ok=True)
    
    # Find all ROC data files for this model
    roc_files = [f for f in os.listdir(results_dir) if f.startswith(f'{model_name.lower().replace(" ", "_")}_roc_data_subject_') and f.endswith('.csv')]
    
    # Loop over each exercise (class label)
    for class_idx, class_label in enumerate(class_labels):
        exercise_data = []  # List to collect data across subjects

        # Loop over each ROC data file
        for roc_file in roc_files:
            # Extract subject ID from file name
            subject_id = roc_file.split('subject_')[1].split('.csv')[0]
            # Read the ROC data
            roc_data_path = os.path.join(results_dir, roc_file)
            roc_data = pd.read_csv(roc_data_path)

            # Check if 'Prob_Class_{class_idx}' exists in the columns
            prob_col_name = f'Prob_Class_{class_idx}'
            if prob_col_name not in roc_data.columns:
                print(f"Probability column '{prob_col_name}' not found in {roc_file}. Skipping.")
                continue

            # Binarize the labels for one-vs-all
            roc_data['Binarized_Label'] = np.where(roc_data['True_Label'] == class_idx, 1, 0)

            # Extract the probability scores for the current class
            roc_data['Score'] = roc_data[prob_col_name]

            # Add subject ID information
            roc_data['SubjectID'] = subject_id

            # Keep only necessary columns
            exercise_df = roc_data[['Score', 'Binarized_Label', 'SubjectID']]

            # Append to the exercise data list
            exercise_data.append(exercise_df)

        # After collecting data from all subjects, combine into a single DataFrame
        if exercise_data:
            all_data = pd.concat(exercise_data, ignore_index=True)
            # Save to CSV
            exercise_csv_path = os.path.join(per_exercise_dir, f'Exercise_{class_idx}_roc_data.csv')
            all_data.to_csv(exercise_csv_path, index=False)
            print(f"ROC data for exercise '{class_label}' saved at '{exercise_csv_path}'")
        else:
            print(f"No data found for exercise '{class_label}'")

def plot_roc_per_exercise(per_exercise_dir, class_labels, output_dir):
    """
    Generates and saves the ROC curve plots for each exercise using the per-exercise ROC data.
    For each exercise, plots ROC curves for each subject in different colors, and computes the mean ROC curve with standard deviation.

    Parameters:
    - per_exercise_dir (str): The directory where the per-exercise ROC data CSV files are stored.
    - class_labels (list): List of class labels (exercise names).
    - output_dir (str): The directory where the ROC plot images will be saved.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist

    for class_idx, class_label in enumerate(class_labels):
        exercise_csv_path = os.path.join(per_exercise_dir, f'Exercise_{class_idx}_roc_data.csv')
        if os.path.exists(exercise_csv_path):
            exercise_data = pd.read_csv(exercise_csv_path)

            # Initialize variables to store TPR, FPR, and AUC for each subject
            tprs = []
            aucs = []
            mean_fpr = np.linspace(0, 1, 100)

            # Prepare to plot
            plt.figure(figsize=(12, 8))

            # Get unique subject IDs
            subject_ids = exercise_data['SubjectID'].unique()
            # Colors for subjects
            colors = plt.get_cmap('tab10', len(subject_ids))

            for idx, subject_id in enumerate(subject_ids):
                subject_data = exercise_data[exercise_data['SubjectID'] == subject_id]
                y_true = subject_data['Binarized_Label']
                y_scores = subject_data['Score']

                if len(np.unique(y_true)) < 2:
                    # Cannot compute ROC curve if only one class is present
                    print(f"Subject {subject_id} for exercise '{class_label}' has only one class.")
                    continue

                fpr, tpr, _ = roc_curve(y_true, y_scores)
                roc_auc = auc(fpr, tpr)

                # Plot per-subject ROC curve
                plt.plot(fpr, tpr, lw=1.5, alpha=0.8,
                         label=f'{subject_id} (AUC = {roc_auc:.2f})', color=colors(idx))

                # Interpolate TPR at mean FPR for mean curve calculation
                interp_tpr = np.interp(mean_fpr, fpr, tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)
                aucs.append(roc_auc)

            # Compute mean and std of TPRs
            if tprs:
                mean_tpr = np.mean(tprs, axis=0)
                std_tpr = np.std(tprs, axis=0)
                mean_tpr[-1] = 1.0  # Ensure TPR ends at 1
                mean_auc = auc(mean_fpr, mean_tpr)
                std_auc = np.std(aucs)

                # Plot mean ROC curve
                plt.plot(mean_fpr, mean_tpr, color='blue', lw=2, alpha=0.9, label=f'Mean ROC')

                # Plot standard deviation as shaded area
                tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
                tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
                plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color='grey', alpha=0.2,
                                 label='± 1 std. dev.')

            else:
                print(f"No valid ROC curves for exercise '{class_label}'")

            # Add chance line
            plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='red', label='Chance')

            # Set plot limits
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])

            # Set axis labels
            plt.xlabel('False Positive Rate', fontsize=16)
            plt.ylabel('True Positive Rate', fontsize=16)

            # Set title with mean AUC and std deviation
            plt.title(
                f'ROC Curves for {class_label}\n(Mean AUC across 4 subjects = {mean_auc:.2f} ± {std_auc:.2f})',
                fontsize=18
            )

            # Add legend
            plt.legend(loc="lower right", fontsize=12)

            # Save the plot to a file
            plot_filename = f'ROC_Curve_{class_label}.png'
            plot_filepath = os.path.join(output_dir, plot_filename)
            plt.savefig(plot_filepath, bbox_inches='tight', dpi=300)
            plt.close()  # Close the figure to free memory
            print(f"ROC plot for exercise '{class_label}' saved at '{plot_filepath}'")
        else:
            print(f"ROC data file not found for exercise '{class_label}'")

if __name__ == "__main__":
    # Define the path to your results directory
    results_dir = './'  # Replace with your actual results directory path

    # Define the model name
    model_name = 'Random Forest'  # Or 'XGBoost'

    # Define the class labels (exercise names)
    class_labels = ['Exercise 0', 'Exercise 1', 'Exercise 2', 'Exercise 3', 'Exercise 4',
                    'Exercise 5', 'Exercise 6 (this is ex 10 fromthe dataset so 3 subjects)', 'Exercise 7', 'Exercise 8', 'Exercise 9']
    
    # Collect ROC data per exercise
    collect_roc_data_per_exercise(
        results_dir=results_dir,
        model_name=model_name,
        class_labels=class_labels
    )

    # Define the path to the per-exercise ROC data directory
    per_exercise_dir = os.path.join(results_dir, f'{model_name.lower().replace(" ", "_")}_per_exercise_roc_data')

    # Define the output directory for the ROC plots
    roc_plots_dir = os.path.join(results_dir, f'{model_name.lower().replace(" ", "_")}_roc_plots')
    os.makedirs(roc_plots_dir, exist_ok=True)

    # Plot and save the ROC curves for each exercise
    plot_roc_per_exercise(per_exercise_dir, class_labels, roc_plots_dir)
