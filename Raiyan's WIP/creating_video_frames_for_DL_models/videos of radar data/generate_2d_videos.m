% Script: generate_filtered_2d_videos_all_subjects.m
% Purpose: Generate videos with filtered and normalized Doppler vs. Intensity plots for all subjects.

% Initialize variables
subjects = {'subject1', 'subject2', 'subject3', 'subject4'};
exercises = 1:10; % Exercises 1 to 10
outputFolder = './videos_2d'; % Folder for output 2D videos
frameRate = 10; % Frames per second for the videos

% Create output directory if not exist
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% Step 0: Calculate global Doppler and Intensity limits across all subjects and exercises
fprintf('Step 0: Calculating global Doppler and Intensity limits...\n');
globalDoppler = [];
globalIntensity = [];

for s = 1:length(subjects)
    subjectDir = fullfile('../../../synced_data/wooutlier/', subjects{s});

    for e = exercises
        fileName = fullfile(subjectDir, sprintf('radar_data%d.mat', e));
        if exist(fileName, 'file')
            dataStruct = load(fileName);
            radarData = dataStruct.radar_data_cropped;

            % Append Doppler and Intensity to global arrays
            globalDoppler = [globalDoppler; radarData(:, 6)];
            globalIntensity = [globalIntensity; radarData(:, 7)];
        else
            warning('File %s does not exist. Skipping exercise %d.', fileName, e);
        end
    end
end

% Calculate global limits
dopplerPercentile = prctile(globalDoppler, [1, 99]);
dopplerMean = mean(globalDoppler);
dopplerStd = std(globalDoppler);
dopplerLimits = [max(dopplerPercentile(1), dopplerMean - 3 * dopplerStd), ...
                 min(dopplerPercentile(2), dopplerMean + 3 * dopplerStd)];

intensityPercentile = prctile(globalIntensity, [1, 99]);
intensityMean = mean(globalIntensity);
intensityStd = std(globalIntensity);
intensityLimits = [max(intensityPercentile(1), intensityMean - 3 * intensityStd), ...
                   min(intensityPercentile(2), intensityMean + 3 * intensityStd)];

fprintf('Filtered Doppler range: [%f, %f]\n', dopplerLimits(1), dopplerLimits(2));
fprintf('Filtered Intensity range: [%f, %f]\n', intensityLimits(1), intensityLimits(2));

% Step 1: Generate videos for all subjects and exercises
fprintf('Step 1: Creating videos with filtered and normalized Doppler vs. Intensity plots...\n');

for s = 1:length(subjects)
    chosenSubject = subjects{s};
    subjectDir = fullfile('../../../synced_data/wooutlier/', chosenSubject);

    for e = exercises
        fileName = fullfile(subjectDir, sprintf('radar_data%d.mat', e));
        if ~exist(fileName, 'file')
            warning('File %s does not exist. Skipping exercise %d.', fileName, e);
            continue;
        end

        dataStruct = load(fileName);
        radarData = dataStruct.radar_data_cropped;
        frames = radarData(:, 1); % Frame numbers
        uniqueFrames = unique(frames);

        % Define video file name
        videoName = sprintf('%s_exercise%d.mp4', chosenSubject, e);
        videoPath = fullfile(outputFolder, videoName);

        % Create video writer
        videoWriter = VideoWriter(videoPath, 'MPEG-4');
        videoWriter.FrameRate = frameRate;
        open(videoWriter);

        % Loop through each frame and plot filtered and normalized Doppler vs. Intensity
        for f = 1:length(uniqueFrames)
            frameNum = uniqueFrames(f);
            frameData = radarData(frames == frameNum, :);
            doppler = frameData(:, 6); % Doppler values
            intensity = frameData(:, 7); % Intensity values

            % Apply global filter
            validIndices = (doppler >= dopplerLimits(1) & doppler <= dopplerLimits(2)) & ...
                           (intensity >= intensityLimits(1) & intensity <= intensityLimits(2));
            dopplerFiltered = doppler(validIndices);
            intensityFiltered = intensity(validIndices);

            % Normalize Doppler and Intensity (within global filtered range)
            dopplerNormalized = (dopplerFiltered - dopplerLimits(1)) / (dopplerLimits(2) - dopplerLimits(1));
            intensityNormalized = (intensityFiltered - intensityLimits(1)) / (intensityLimits(2) - intensityLimits(1));

            % Create 2D scatter plot
            figure('Visible', 'off'); % Create invisible figure
            scatter(intensityNormalized, dopplerNormalized, 15, 'filled');

            % Set axis properties
            grid off;
            axis tight;
            xlim([0, 1]); % Normalized Doppler range
            ylim([0, 1]); % Normalized Intensity range

            % Enable ticks without labels
            set(gca, 'XColor', 'k', 'YColor', 'k', 'FontSize', 10); % Enable ticks

            % Capture the frame for the video
            frame = getframe(gcf);
            writeVideo(videoWriter, frame);
            close(gcf); % Close the figure to avoid memory issues
        end

        % Close video writer
        close(videoWriter);
        fprintf('Created video: %s\n', videoName);
    end
end

fprintf('Videos created successfully in %s.\n', outputFolder);
