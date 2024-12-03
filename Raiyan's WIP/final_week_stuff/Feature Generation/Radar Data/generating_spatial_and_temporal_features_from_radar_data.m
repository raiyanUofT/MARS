% Data Loading

% Initialize variables
subjects = {'subject1', 'subject2', 'subject3', 'subject4'};
exercises = 1:10; % Exercises 1 to 10
exerciseNames = {
    'Left upper limb extension', 'Right upper limb extension', 'Both upper limb extension', ...
    'Left front lunge', 'Right front lunge', 'Squat', ...
    'Left side lunge', 'Right side lunge', 'Left limb extension', 'Right limb extension'};

% Initialize a cell array to store the data
allData = {};

% Loop over subjects
for s = 1:length(subjects)
    subjectDir = fullfile('./../../../../../synced_data/woutlier/', subjects{s});
    
    % Adjust exercises for subject4
    if strcmp(subjects{s}, 'subject4')
        exercisesToProcess = 1:9; % Only first 9 exercises
    else
        exercisesToProcess = exercises;
    end
    
    % Loop over exercises
    for e = exercisesToProcess
        % Construct the file name
        fileName = fullfile(subjectDir, sprintf('radar_data%d.mat', e));
        
        % Check if file exists
        if exist(fileName, 'file')
            % Load the radar data file
            dataStruct = load(fileName);
            radarData = dataStruct.radar_data_cropped;
            
            % Store data with subject and exercise identifiers
            allData{end+1} = struct('subject', subjects{s}, 'exercise', e, 'data', radarData);
        else
            fprintf('File %s does not exist.\n', fileName);
        end
    end
end

% Preprocessing and Voxelization

% Define voxelization parameters
voxelSize = 0.5; % Adjust as needed
globalMinCoords = [inf, inf, inf];
globalMaxCoords = [-inf, -inf, -inf];

% First pass to find global coordinate bounds
for i = 1:length(allData)
    radarData = allData{i}.data;
    
    % Extract coordinates
    x = radarData(:, 3);
    y = radarData(:, 4);
    z = radarData(:, 5);
    
    % Update global min and max coordinates
    globalMinCoords = min([globalMinCoords; [min(x), min(y), min(z)]], [], 1);
    globalMaxCoords = max([globalMaxCoords; [max(x), max(y), max(z)]], [], 1);
end

% Compute grid size
gridSize = ceil((globalMaxCoords - globalMinCoords) / voxelSize);

% Now process each data entry
processedData = {};

% Define variable names for features
variableNames = {'FrameNum', 'NumPoints', 'MeanX', 'StdX', 'SkewX', ...
                 'KurtX', 'MeanY', 'StdY', 'SkewY', 'KurtY', 'MeanZ', ...
                 'StdZ', 'SkewZ', 'KurtZ', 'MeanDoppler', 'StdDoppler', ...
                 'MeanIntensity', 'StdIntensity', 'RangeX', 'RangeY', ...
                 'RangeZ', 'NumOccupiedVoxels', 'VoxelDensity', 'MaxIntensity', ...
                 'MinIntensity', 'MeanRadialDist', 'StdRadialDist', 'VoxelEntropy', ...
                 'PCA_DirectionX', 'PCA_DirectionY', 'PCA_DirectionZ', ...
                 'ExplainedVariance', 'MeanGradX', 'MeanGradY', 'MeanGradZ', ...
                 'CovXY', 'CovXZ', 'CovYZ', 'CorrXY', 'CorrXZ', 'CorrYZ', ...
                 'EigVal1', 'EigVal2', 'EigVal3', 'EigRatio1', 'EigRatio2', ...
                 'SkewDoppler', 'KurtDoppler', 'SkewIntensity', 'KurtIntensity', ...
                 'MaxDoppler', 'MinDoppler', 'MaxRadialDist', 'MinRadialDist', ...
                 'TotalIntensity', 'ConvexHullVolume', 'DopplerEntropy', 'IntensityEntropy'};

% Add temporal variable names
temporalVariableNames = {'DeltaNumPoints', 'DeltaMeanX', 'DeltaStdX', 'DeltaSkewX', ...
                         'DeltaKurtX', 'DeltaMeanY', 'DeltaStdY', 'DeltaSkewY', ...
                         'DeltaKurtY', 'DeltaMeanZ', 'DeltaStdZ', 'DeltaSkewZ', ...
                         'DeltaKurtZ', 'DeltaMeanDoppler', 'DeltaStdDoppler', ...
                         'DeltaMeanIntensity', 'DeltaStdIntensity', 'DeltaRangeX', 'DeltaRangeY', ...
                         'DeltaRangeZ', 'DeltaNumOccupiedVoxels', 'DeltaVoxelDensity', 'DeltaMaxIntensity', ...
                         'DeltaMinIntensity', 'DeltaMeanRadialDist', 'DeltaStdRadialDist', 'DeltaVoxelEntropy', ...
                         'DeltaPCA_DirectionX', 'DeltaPCA_DirectionY', 'DeltaPCA_DirectionZ', ...
                         'DeltaExplainedVariance', 'DeltaMeanGradX', 'DeltaMeanGradY', 'DeltaMeanGradZ', ...
                         'DeltaCovXY', 'DeltaCovXZ', 'DeltaCovYZ', 'DeltaCorrXY', 'DeltaCorrXZ', 'DeltaCorrYZ', ...
                         'DeltaEigVal1', 'DeltaEigVal2', 'DeltaEigVal3', 'DeltaEigRatio1', 'DeltaEigRatio2', ...
                         'DeltaSkewDoppler', 'DeltaKurtDoppler', 'DeltaSkewIntensity', 'DeltaKurtIntensity', ...
                         'DeltaMaxDoppler', 'DeltaMinDoppler', 'DeltaMaxRadialDist', 'DeltaMinRadialDist', ...
                         'DeltaTotalIntensity', 'DeltaConvexHullVolume', 'DeltaDopplerEntropy', 'DeltaIntensityEntropy'};

% Combine variable names
variableNames = [variableNames, temporalVariableNames];

% Process each data entry and calculate features
for i = 1:length(allData)
    radarData = allData{i}.data;
    frames = radarData(:, 1); % Frame numbers
    uniqueFrames = unique(frames);
    
    % Initialize storage for features
    featuresList = [];
    
    % Initialize previous frame features (set to NaN initially)
    prevFeatureVector = NaN(1, length(variableNames) - length(temporalVariableNames));
    
    for f = 1:length(uniqueFrames)
        frameNum = uniqueFrames(f);
        frameData = radarData(frames == frameNum, :);
        
        % Extract coordinates and attributes
        x = frameData(:, 3);
        y = frameData(:, 4);
        z = frameData(:, 5);
        doppler = frameData(:, 6);
        intensity = frameData(:, 7);
        
        % Number of points
        numPoints = size(frameData, 1);
        
        % Voxelization
        points = [x, y, z];
        voxelIndices = floor((points - globalMinCoords) / voxelSize) + 1;
        % Ensure indices are within grid size
        voxelIndices = min(max(voxelIndices, 1), gridSize);
        linearIndices = sub2ind(gridSize, voxelIndices(:,1), voxelIndices(:,2), voxelIndices(:,3));
        uniqueVoxels = unique(linearIndices);
        numOccupiedVoxels = numel(uniqueVoxels);
        
        % Additional features from voxel data
        voxelDensity = numPoints / numOccupiedVoxels; % Point density per voxel
        maxIntensity = max(intensity); % Maximum intensity within the frame
        minIntensity = min(intensity); % Minimum intensity
        
        % Spatial distribution metrics
        rangeX = max(x) - min(x);
        rangeY = max(y) - min(y);
        rangeZ = max(z) - min(z);
        
        % Statistical features
        meanX = mean(x);
        stdX = std(x);
        skewX = skewness(x);
        kurtX = kurtosis(x);
        
        meanY = mean(y);
        stdY = std(y);
        skewY = skewness(y);
        kurtY = kurtosis(y);
        
        meanZ = mean(z);
        stdZ = std(z);
        skewZ = skewness(z);
        kurtZ = kurtosis(z);
        
        % Doppler features
        meanDoppler = mean(doppler);
        stdDoppler = std(doppler);
        
        % Intensity features
        meanIntensity = mean(intensity);
        stdIntensity = std(intensity);
        
        % Radial distance from radar center
        radialDistances = sqrt(x.^2 + y.^2 + z.^2);
        meanRadialDist = mean(radialDistances);
        stdRadialDist = std(radialDistances);
        
        % Voxel entropy for occupancy and intensity
        voxelEntropy = -sum((intensity ./ sum(intensity)) .* log2(intensity ./ sum(intensity) + eps));
        
        % PCA on voxelized data for principal direction of movement
        if numPoints > 1
            [coeff, ~, latent] = pca([x, y, z]);
            pcaPrimaryDirection = coeff(:,1); % Principal component direction
            explainedVariance = latent(1) / sum(latent); % Variance explained by primary direction
        else
            pcaPrimaryDirection = [NaN; NaN; NaN];
            explainedVariance = NaN;
        end
        
        % Directional gradient features between neighboring points
        gradX = gradient(x);
        gradY = gradient(y);
        gradZ = gradient(z);
        
        % Calculate the mean of the gradients to capture directional flow
        meanGradX = mean(gradX);
        meanGradY = mean(gradY);
        meanGradZ = mean(gradZ);
        
        % Covariance and Correlation between X, Y, Z
        if numPoints > 1
            covMatrix = cov([x, y, z]);
            covXY = covMatrix(1,2);
            covXZ = covMatrix(1,3);
            covYZ = covMatrix(2,3);
            
            corrMatrix = corrcoef([x, y, z]);
            corrXY = corrMatrix(1,2);
            corrXZ = corrMatrix(1,3);
            corrYZ = corrMatrix(2,3);
        else
            covXY = NaN;
            covXZ = NaN;
            covYZ = NaN;
            corrXY = NaN;
            corrXZ = NaN;
            corrYZ = NaN;
        end
        
        % Eigenvalues of Covariance Matrix
        if numPoints > 1
            [~, eigValsMatrix] = eig(covMatrix);
            eigVals = diag(eigValsMatrix);
            % Sort eigenvalues in descending order
            [eigVals, ~] = sort(eigVals, 'descend');
            eigVal1 = eigVals(1);
            eigVal2 = eigVals(2);
            eigVal3 = eigVals(3);
        else
            eigVal1 = NaN;
            eigVal2 = NaN;
            eigVal3 = NaN;
        end
        
        % Ratios of Eigenvalues
        if numPoints > 1 && all(~isnan(eigVals))
            eigRatio1 = eigVal1 / eigVal2;
            eigRatio2 = eigVal2 / eigVal3;
        else
            eigRatio1 = NaN;
            eigRatio2 = NaN;
        end
        
        % Skewness and Kurtosis of Doppler and Intensity
        if numPoints > 1
            skewDoppler = skewness(doppler);
            kurtDoppler = kurtosis(doppler);
            skewIntensity = skewness(intensity);
            kurtIntensity = kurtosis(intensity);
        else
            skewDoppler = NaN;
            kurtDoppler = NaN;
            skewIntensity = NaN;
            kurtIntensity = NaN;
        end
        
        % Max and Min Doppler
        maxDoppler = max(doppler);
        minDoppler = min(doppler);
        
        % Max and Min Radial Distance
        maxRadialDist = max(radialDistances);
        minRadialDist = min(radialDistances);
        
        % Total Intensity
        totalIntensity = sum(intensity);
        
        % Convex Hull Volume
        if numPoints >= 4
            try
                [~, volume] = convhull(x, y, z);
            catch
                volume = NaN;
            end
        else
            volume = NaN;
        end
        
        % Entropy of Doppler and Intensity Distributions
        % For Doppler
        dopplerHist = histcounts(doppler, 'Normalization', 'probability');
        dopplerEntropy = -sum(dopplerHist .* log2(dopplerHist + eps));
        
        % For Intensity
        intensityHist = histcounts(intensity, 'Normalization', 'probability');
        intensityEntropy = -sum(intensityHist .* log2(intensityHist + eps));
        
        % Compile features into a vector
        featureVector = [frameNum, numPoints, meanX, stdX, skewX, kurtX, ...
                         meanY, stdY, skewY, kurtY, meanZ, stdZ, skewZ, kurtZ, ...
                         meanDoppler, stdDoppler, meanIntensity, stdIntensity, ...
                         rangeX, rangeY, rangeZ, numOccupiedVoxels, voxelDensity, ...
                         maxIntensity, minIntensity, meanRadialDist, stdRadialDist, voxelEntropy, ...
                         pcaPrimaryDirection(1), pcaPrimaryDirection(2), pcaPrimaryDirection(3), ...
                         explainedVariance, meanGradX, meanGradY, meanGradZ, ...
                         covXY, covXZ, covYZ, corrXY, corrXZ, corrYZ, ...
                         eigVal1, eigVal2, eigVal3, eigRatio1, eigRatio2, ...
                         skewDoppler, kurtDoppler, skewIntensity, kurtIntensity, ...
                         maxDoppler, minDoppler, maxRadialDist, minRadialDist, ...
                         totalIntensity, volume, dopplerEntropy, intensityEntropy];
        
        % Temporal Features: Differences with previous frame
        if f > 1
            deltaFeatures = featureVector(2:end) - prevFeatureVector(2:end); % Exclude FrameNum
        else
            deltaFeatures = NaN(1, length(featureVector) - 1);
        end
        
        % Update previous feature vector
        prevFeatureVector = featureVector;
        
        % Append temporal features to feature vector
        featureVector = [featureVector, deltaFeatures];
        
        % Append to features list
        featuresList = [featuresList; featureVector];
    end
    
    % Combine variable names and ensure they match the feature vector length
    if size(featuresList, 2) ~= length(variableNames)
        error('Feature vector length does not match variable names length.');
    end
    
    % Convert featuresList to a table with variable names
    featuresTable = array2table(featuresList, 'VariableNames', variableNames);
    
    % Add labels for subject and exercise
    exerciseIndex = allData{i}.exercise;
    subjectID = allData{i}.subject;
    
    % Create labels for each frame
    numFrames = size(featuresTable, 1);
    exerciseLabels = repmat({exerciseNames{exerciseIndex}}, numFrames, 1);
    subjectLabels = repmat({subjectID}, numFrames, 1);
    
    % Add labels to the featuresTable
    featuresTable.ExerciseLabel = exerciseLabels;
    featuresTable.SubjectID = subjectLabels;
    
    % Store processed data with features table
    processedData{end+1} = struct('subject', subjectID, 'exercise', exerciseIndex, 'featuresTable', featuresTable);
end

% Data Saving to .mat
save('spatial_and_temporal_features_from_radar_data.mat', 'processedData', '-v7.3');

% Data Saving to .csv

% Initialize an empty table to hold all data
allFeaturesTable = table();

% Concatenate tables from each entry
for i = 1:length(processedData)
    currentTable = processedData{i}.featuresTable;
    allFeaturesTable = [allFeaturesTable; currentTable];
end

% Save to CSV
writetable(allFeaturesTable, 'spatial_and_temporal_features_from_radar_data.csv');
