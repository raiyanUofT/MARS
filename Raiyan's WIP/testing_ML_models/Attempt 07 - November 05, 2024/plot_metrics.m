% Load the CSV file with 'VariableNamingRule' set to 'preserve' to keep original column headers
data = readtable('Results/metrics.csv', 'VariableNamingRule', 'preserve');

% Define acronyms and full names for models
model_acronyms = {'LR', 'DT', 'KNN', 'SVM', 'NB', 'RF', 'XGB'};  % Short labels for each model
full_model_names = {'Logistic Regression', 'Decision Tree', 'K-Nearest Neighbors', ...
                    'Support Vector Machine', 'Naive Bayes', 'Random Forest', 'XGBoost'};  % Full names for legend
legend_labels = strcat(model_acronyms, ' - ', full_model_names);  % Combine acronyms and full names

% Extract specific metrics
accuracy = data.Accuracy;
precision = data.Precision;
recall = data.Recall;
f1_score = data.("F1 Score");  % Use double quotes for names with spaces
specificity = data.Specificity;

% Define metrics and their values in a cell array for easy iteration
metrics = {'Accuracy', accuracy; 'Precision', precision; 'Recall', recall; 'F1 Score', f1_score; 'Specificity', specificity};

% Define a warm, high-contrast color scheme for the bar plots
warm_color_scheme = [
    1.0, 0.9, 0.8;  % Very light peach
    1.0, 0.7, 0.5;  % Light peach-orange
    1.0, 0.5, 0.3;  % Medium peach
    1.0, 0.3, 0.1;  % Dark peach
    0.8, 0.1, 0.1;  % Deep red
    0.6, 0.0, 0.0;  % Dark red
    0.4, 0.0, 0.0   % Darkest red
];

% Set up figure with a 3x2 grid
figure('Position', [100, 100, 1200, 800]);

for i = 1:5
    % Create subplot in a 3x2 layout
    subplot(3, 2, i);
    
    % Plot each model as a separate bar with color coding
    hold on;
    for j = 1:numel(model_acronyms)
        bar(j, metrics{i, 2}(j), 'FaceColor', warm_color_scheme(j, :), 'BarWidth', 0.8);
    end
    hold off;
    
    % Customize plot
    title([metrics{i, 1} ' Comparison'], 'FontSize', 14, 'FontWeight', 'bold');
    ylabel(metrics{i, 1}, 'FontSize', 12, 'FontWeight', 'bold');
    ylim([0, 100]);
    set(gca, 'XTick', 1:numel(model_acronyms), 'XTickLabel', model_acronyms);  % Set x-axis with acronyms
end

% Add an overall title for the plot
sgtitle('Model Performance Comparison', 'FontSize', 16, 'FontWeight', 'bold');

% Create the legend in the sixth subplot
subplot(3, 2, 6);
hold on;
for j = 1:numel(full_model_names)
    % Plot invisible bars just for the legend
    plot(NaN, NaN, 's', 'MarkerFaceColor', warm_color_scheme(j, :), 'MarkerEdgeColor', warm_color_scheme(j, :), 'MarkerSize', 8);
end
hold off;
axis off;  % Turn off the axis for the legend area
legend(legend_labels, 'Location', 'north', 'FontSize', 12);
title('Models', 'FontSize', 14, 'FontWeight', 'bold');

% Set background color and save
set(gcf, 'Color', 'w');
saveas(gcf, 'Results/Images for Project/metrics_comparison.png');
