% Script: generate_videos.m
% Generate 39 videos, one for each exercise per subject, with radar data and visible axes.

% Initialize variables
subjects = {'subject1', 'subject2', 'subject3', 'subject4'};
exercises = 1:10; % Exercises 1 to 10
outputFolder = './videos_3d'; % Folder for initial videos
frameRate = 10; % Frames per second for the videos

% Create output directory if not exist
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% Step 0: Calculate global axis limits
fprintf('Step 0: Calculating global axis limits...\n');
globalMin = [inf, inf, inf];
globalMax = [-inf, -inf, -inf];

for s = 1:length(subjects)
    subjectDir = fullfile('../../../synced_data/woutlier/', subjects{s});
    
    if strcmp(subjects{s}, 'subject4')
        exercisesToProcess = 1:9; % Only first 9 exercises
    else
        exercisesToProcess = exercises;
    end
    
    for e = exercisesToProcess
        fileName = fullfile(subjectDir, sprintf('radar_data%d.mat', e));
        if exist(fileName, 'file')
            dataStruct = load(fileName);
            radarData = dataStruct.radar_data_cropped;
            globalMin = min(globalMin, [min(radarData(:, 3)), min(radarData(:, 4)), min(radarData(:, 5))]);
            globalMax = max(globalMax, [max(radarData(:, 3)), max(radarData(:, 4)), max(radarData(:, 5))]);
        end
    end
end
fprintf('Global axis limits calculated: X[%f, %f], Y[%f, %f], Z[%f, %f]\n', ...
    globalMin(1), globalMax(1), globalMin(2), globalMax(2), globalMin(3), globalMax(3));

% Step 1: Create 39 videos
fprintf('Step 1: Creating 39 videos...\n');
for s = 1:length(subjects)
    subjectDir = fullfile('../../../synced_data/woutlier/', subjects{s});
    
    if strcmp(subjects{s}, 'subject4')
        exercisesToProcess = 1:9; % Only first 9 exercises
    else
        exercisesToProcess = exercises;
    end
    
    for e = exercisesToProcess
        fileName = fullfile(subjectDir, sprintf('radar_data%d.mat', e));
        if exist(fileName, 'file')
            dataStruct = load(fileName);
            radarData = dataStruct.radar_data_cropped;
            frames = radarData(:, 1); % Frame numbers
            uniqueFrames = unique(frames);
            
            % Define video file name
            videoName = sprintf('%s_exercise%d.mp4', subjects{s}, e);
            videoPath = fullfile(outputFolder, videoName);
            
            % Create video writer
            videoWriter = VideoWriter(videoPath, 'MPEG-4');
            videoWriter.FrameRate = frameRate;
            open(videoWriter);
            
            % Loop through each frame and plot radar points
            for f = 1:length(uniqueFrames)
                frameNum = uniqueFrames(f);
                frameData = radarData(frames == frameNum, :);
                x = frameData(:, 3);
                y = frameData(:, 4);
                z = frameData(:, 5);
                
                % Plot radar points in 3D space
                figure('Visible', 'off'); % Create invisible figure
                scatter3(x, y, z, 15, 'filled');
                
                % Set axis properties
                grid off;
                axis tight;
                xlim([globalMin(1), globalMax(1)]);
                ylim([globalMin(2), globalMax(2)]);
                zlim([globalMin(3), globalMax(3)]);
                
                % Enable tick marks and values without labels
                set(gca, 'XColor', 'k', 'YColor', 'k', 'ZColor', 'k'); % Make axis visible
                set(gca, 'FontSize', 10); % Adjust font size for better visibility
                
                % Set consistent view angle
                view([45, 30]); % Azimuth 45°, Elevation 30°
                
                % Capture the frame for the video
                frame = getframe(gcf);
                writeVideo(videoWriter, frame);
                close(gcf); % Close the figure to avoid memory issues
            end
            
            % Close video writer
            close(videoWriter);
            fprintf('Created video: %s\n', videoName);
        else
            fprintf('File %s does not exist.\n', fileName);
        end
    end
end
fprintf('39 videos created successfully in %s.\n', outputFolder);
