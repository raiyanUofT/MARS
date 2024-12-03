% Load the metrics.csv file
data = readtable('Results/metrics.csv');

% Initialize two cell arrays to store the formatted tables
formatted_table_main = cell(0, 3);  % For models except Random Forest and XGBoost
formatted_table_rf_xgb = cell(0, 3);  % For Random Forest and XGBoost
table_columns = {'Model', 'Hyperparameter', 'Value'};

% Loop over each row in the original table to expand hyperparameters
for i = 1:height(data)
    model_name = data.Model{i};
    params_str = data.("BestParameters"){i};
    
    % Convert parameter string to a structured format using regex for splitting
    params_str = regexprep(params_str, '[{}'']', '');  % Remove braces and quotes
    param_pairs = strsplit(params_str, ', ');          % Split by commas to get pairs
    
    % Insert model name for the first row only
    first_row = true;
    
    % Process each parameter pair
    for j = 1:length(param_pairs)
        param_pair = strsplit(param_pairs{j}, ': ');
        param_name = strtrim(param_pair{1});
        param_value = strtrim(param_pair{2});
        
        % Add model name only in the first row for each model's parameters
        if first_row
            row = {model_name, param_name, param_value};
            first_row = false;  % Only display model name once
        else
            row = {'', param_name, param_value};
        end
        
        % Append row to the appropriate table
        if strcmp(model_name, 'Random Forest') || strcmp(model_name, 'XGBoost')
            formatted_table_rf_xgb = [formatted_table_rf_xgb; row];
        else
            formatted_table_main = [formatted_table_main; row];
        end
    end
end

% Convert to table format for display
formatted_table_main = cell2table(formatted_table_main, 'VariableNames', table_columns);
formatted_table_rf_xgb = cell2table(formatted_table_rf_xgb, 'VariableNames', table_columns);

% Plotting the main table as an image with consistent styling
fig1 = figure('Position', [100, 100, 1200, 600], 'Color', 'w');
uitable('Data', formatted_table_main{:,:}, ...
        'ColumnName', formatted_table_main.Properties.VariableNames, ...
        'RowName', [], ...
        'Position', [20 20 1150 550], ...
        'ColumnWidth', {150, 350, 500}, ...
        'FontSize', 10, ...
        'FontName', 'Arial', 'BackgroundColor', [1 1 1]);  % Consistent font and color

% Save the main table as an image
saveas(fig1, 'Results/Images for Project/main_hyperparameters_table.png');
close(fig1);

% Plotting the Random Forest and XGBoost table as an image with consistent styling
fig2 = figure('Position', [100, 100, 1200, 400], 'Color', 'w');
uitable('Data', formatted_table_rf_xgb{:,:}, ...
        'ColumnName', formatted_table_rf_xgb.Properties.VariableNames, ...
        'RowName', [], ...
        'Position', [20 20 1150 350], ...
        'ColumnWidth', {150, 350, 500}, ...
        'FontSize', 10, ...
        'FontName', 'Arial', 'BackgroundColor', [1 1 1]);  % Consistent font and color

% Save the Random Forest and XGBoost table as an image
saveas(fig2, 'Results/Images for Project/rf_xgb_hyperparameters_table.png');
close(fig2);
