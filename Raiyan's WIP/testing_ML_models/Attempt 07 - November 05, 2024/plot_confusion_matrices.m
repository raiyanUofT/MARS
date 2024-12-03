% Define file paths for each model's confusion matrix within the Results directory
file_paths = {
    'Results/decision_tree_confusion_matrix.csv', ...
    'Results/k-nearest_neighbors_confusion_matrix.csv', ...
    'Results/logistic_regression_confusion_matrix.csv', ...
    'Results/naive_bayes_confusion_matrix.csv', ...
    'Results/random_forest_confusion_matrix.csv', ...
    'Results/support_vector_machine_confusion_matrix.csv', ...
    'Results/xgboost_confusion_matrix.csv'
};

% Define model names for titles
model_names = {
    'Decision Tree', 'K-Nearest Neighbors', 'Logistic Regression', ...
    'Naive Bayes', 'Random Forest', 'Support Vector Machine', 'XGBoost'
};

% Set up a 3x3 grid for subplots
figure('Position', [100, 100, 1000, 800]);

% Define a high-contrast color scheme for the confusion matrices
custom_colormap = [
    1.0, 0.9, 0.8;  % Very light peach
    1.0, 0.7, 0.5;  % Light peach-orange
    1.0, 0.5, 0.3;  % Medium peach
    1.0, 0.3, 0.1;  % Dark peach
    0.8, 0.1, 0.1;  % Deep red
    0.6, 0.0, 0.0;  % Dark red
    0.3, 0.0, 0.0   % Very dark red
];

colormap(custom_colormap);  % Apply the custom high-contrast colormap

for i = 1:length(file_paths)
    % Load confusion matrix from CSV file, skipping the first row (header)
    matrix = readmatrix(file_paths{i}, 'NumHeaderLines', 1);  % Skips the first row
    
    % Create subplot for each confusion matrix
    subplot(3, 3, i);
    imagesc(matrix);  % Display the matrix as a heatmap
    colorbar;  % Add a colorbar for reference
    
    % Label axes and set title
    title(model_names{i}, 'FontSize', 12, 'FontWeight', 'bold');
    xlabel('Predicted Label', 'FontSize', 10, 'FontWeight', 'bold');
    ylabel('True Label', 'FontSize', 10, 'FontWeight', 'bold');
    
    % Set axis ticks for matrix dimensions
    axis equal;
    xticks(1:size(matrix, 2));
    yticks(1:size(matrix, 1));
    set(gca, 'XTickLabelRotation', 0);  % Set x-axis labels to horizontal
end

% Add an overarching title for the figure
sgtitle('Confusion Matrices for Different Models', 'FontSize', 16, 'FontWeight', 'bold');

% Set background color to white and save the figure
set(gcf, 'Color', 'w');
saveas(gcf, 'Results/Images for Project/confusion_matrices_comparison.png');
