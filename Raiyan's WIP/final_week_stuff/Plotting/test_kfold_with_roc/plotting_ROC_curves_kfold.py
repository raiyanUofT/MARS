import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def collect_roc_data_per_exercise(results_dir, model_name, num_folds, class_labels):
    """
    Collects ROC data per exercise (class label) from all folds and saves them into separate CSV files.

    Parameters:
    - results_dir (str): The directory where the results are stored.
    - model_name (str): The model name whose ROC data to process.
    - num_folds (int): The number of folds used in cross-validation.
    - class_labels (list): List of class labels (exercise names).
    """
    # Create a directory to store the per-exercise ROC data
    per_exercise_dir = os.path.join(results_dir, f'{model_name.lower().replace(" ", "_")}_per_exercise_roc_data')
    os.makedirs(per_exercise_dir, exist_ok=True)

    # Loop over each exercise (class label)
    for class_idx in range(len(class_labels)):
        exercise_data = []  # List to collect data across folds

        # Loop over each fold
        for fold_idx in range(1, num_folds + 1):
            # Path to the ROC data file for this fold
            roc_data_path = os.path.join(
                results_dir,
                f'fold_{fold_idx}',
                f'{model_name.lower().replace(" ", "_")}_roc_data_fold_{fold_idx}.csv'
            )
            if os.path.exists(roc_data_path):
                # Read the ROC data
                roc_data = pd.read_csv(roc_data_path)

                # Add fold information
                roc_data['Fold'] = fold_idx

                # Append to the exercise data list
                exercise_data.append(roc_data)
            else:
                print(f"ROC data file not found: {roc_data_path}")

        # After collecting data from all folds, combine into a single DataFrame
        if exercise_data:
            all_data = pd.concat(exercise_data, ignore_index=True)

            # Binarize the labels for one-vs-all
            all_data['Binarized_Label'] = np.where(all_data['True_Label'] == class_idx, 1, 0)

            # Extract the probability scores for the current class
            prob_class_idx = class_idx

            all_data['Score'] = all_data[f'Prob_Class_{prob_class_idx}']

            # Keep only necessary columns
            exercise_df = all_data[['Score', 'Binarized_Label', 'Fold']]

            # Save to CSV without including the exercise name
            exercise_csv_path = os.path.join(per_exercise_dir, f'Exercise_{class_idx}_roc_data.csv')
            exercise_df.to_csv(exercise_csv_path, index=False)
            print(f"ROC data for exercise '{class_labels[class_idx]}' saved at '{exercise_csv_path}'")
        else:
            print(f"No data found for exercise '{class_labels[class_idx]}'")

def plot_roc_per_exercise(per_exercise_dir, class_labels, output_dir, num_folds):
    """
    Generates and saves the ROC curve plots for each exercise using the per-exercise ROC data.
    For each exercise, plots ROC curves for each fold, and the mean ROC curve with std dev.

    Parameters:
    - per_exercise_dir (str): The directory where the per-exercise ROC data CSV files are stored.
    - class_labels (list): List of class labels (exercise names).
    - output_dir (str): The directory where the ROC plot images will be saved.
    - num_folds (int): The number of folds used in cross-validation.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist

    for class_idx in range(len(class_labels)):
        exercise_csv_path = os.path.join(per_exercise_dir, f'Exercise_{class_idx}_roc_data.csv')
        if os.path.exists(exercise_csv_path):
            exercise_data = pd.read_csv(exercise_csv_path)

            # Initialize variables to store TPR, FPR, and AUC for each fold
            tprs = []
            aucs = []
            mean_fpr = np.linspace(0, 1, 100)

            # Prepare to plot
            plt.figure(figsize=(10, 8))

            # Colors for folds
            colors = plt.cm.get_cmap('tab10', num_folds)

            for fold_idx in range(1, num_folds + 1):
                fold_data = exercise_data[exercise_data['Fold'] == fold_idx]
                y_true = fold_data['Binarized_Label']
                y_scores = fold_data['Score']

                if len(np.unique(y_true)) < 2:
                    # Cannot compute ROC curve if only one class is present
                    print(f"Fold {fold_idx} for exercise '{class_labels[class_idx]}' has only one class.")
                    continue

                fpr, tpr, _ = roc_curve(y_true, y_scores)
                roc_auc = auc(fpr, tpr)
                # Interpolate TPR at mean FPR
                interp_tpr = np.interp(mean_fpr, fpr, tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)
                aucs.append(roc_auc)

                # Plot per-fold ROC curve
                plt.plot(fpr, tpr, lw=1, alpha=0.6,
                         label=f'Fold {fold_idx}', color=colors(fold_idx - 1))

            # Compute mean and std of TPRs
            if tprs:
                mean_tpr = np.mean(tprs, axis=0)
                std_tpr = np.std(tprs, axis=0)
                mean_tpr[-1] = 1.0  # Ensure TPR ends at 1
                mean_auc = auc(mean_fpr, mean_tpr)
                std_auc = np.std(aucs)

                # Plot mean ROC curve
                plt.plot(mean_fpr, mean_tpr, color='blue', lw=2, alpha=0.8, label='Mean ROC')

                # Plot std dev around mean ROC
                tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
                tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
                plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color='grey', alpha=0.2,
                                 label='± 1 std. dev.')
            else:
                print(f"No valid ROC curves for exercise '{class_labels[class_idx]}'")

            # Add chance line
            plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='red', label='Chance')

            # Set plot limits
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            
            # Set axis labels
            plt.xlabel('False Positive Rate', fontsize=14)
            plt.ylabel('True Positive Rate', fontsize=14)
            
            # Set title with mean AUC and std deviation
            plt.title(
                f'ROC Curve for {class_labels[class_idx]} (Mean AUC = {mean_auc:.2f} ± {std_auc:.2f})\n'
                f'(Mean across {num_folds} folds)',
                fontsize=16
            )

            # Add legend
            plt.legend(loc="lower right", fontsize=12)

            # Save the plot to a file
            plot_filename = f'ROC_Curve_Exercise_{class_idx}.png'
            plot_filepath = os.path.join(output_dir, plot_filename)
            plt.savefig(plot_filepath)
            plt.close()  # Close the figure to free memory
            print(f"ROC plot for exercise '{class_labels[class_idx]}' saved at '{plot_filepath}'")
        else:
            print(f"ROC data file not found for exercise '{class_labels[class_idx]}'")

if __name__ == "__main__":
    # Define the path to your results directory
    results_dir = './'  # Replace with your actual results directory path

    # Define the model name
    model_name = 'Random Forest'  # Replace with the model name you used

    # Define the number of folds
    num_folds = 10  # Adjust according to your cross-validation setup

    # Define the class labels (exercise names)
    class_labels = ['Exercise0', 'Exercise1', 'Exercise2', 'Exercise3', 'Exercise4',
                    'Exercise5', 'Exercise6', 'Exercise7', 'Exercise8', 'Exercise9']

    # Collect ROC data per exercise
    collect_roc_data_per_exercise(
        results_dir=results_dir,
        model_name=model_name,
        num_folds=num_folds,
        class_labels=class_labels
    )

    # Define the path to the per-exercise ROC data directory
    per_exercise_dir = os.path.join(results_dir, f'{model_name.lower().replace(" ", "_")}_per_exercise_roc_data')

    # Define the output directory for the ROC plots
    roc_plots_dir = os.path.join(results_dir, f'{model_name.lower().replace(" ", "_")}_roc_plots')
    os.makedirs(roc_plots_dir, exist_ok=True)

    # Plot and save the ROC curves for each exercise
    plot_roc_per_exercise(per_exercise_dir, class_labels, roc_plots_dir, num_folds)
