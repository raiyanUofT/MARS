% Folder containing segmented videos
inputFolder = './segments_2d';
outputFolder = './segments_2d_resampled';

% Create output directory if it does not exist
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% Target frame count for each video
targetFrameCount = 400;

% Get list of all video files in the folder
videoFiles = dir(fullfile(inputFolder, '*.mp4'));

% Process each video file
fprintf('Resampling video segments to have exactly %d frames...\n', targetFrameCount);

for i = 1:length(videoFiles)
    videoPath = fullfile(videoFiles(i).folder, videoFiles(i).name);
    videoName = videoFiles(i).name;
    
    % Read video
    videoReader = VideoReader(videoPath);
    frameRate = videoReader.FrameRate; % Preserve original frame rate
    totalFrames = floor(videoReader.Duration * frameRate);
    
    % Determine resampling step
    if totalFrames <= targetFrameCount
        fprintf('Video %s already has %d frames. No resampling required.\n', videoName, totalFrames);
        copyfile(videoPath, fullfile(outputFolder, videoName)); % Copy video as is
        continue;
    end
    
    % Downsample frames
    downsampleStep = totalFrames / targetFrameCount;
    selectedFrames = round(1:downsampleStep:totalFrames);
    
    % Create new video writer
    outputVideoPath = fullfile(outputFolder, videoName);
    videoWriter = VideoWriter(outputVideoPath, 'MPEG-4');
    videoWriter.FrameRate = frameRate;
    open(videoWriter);
    
    % Read and write downsampled frames
    frameIndex = 1;
    while hasFrame(videoReader)
        frame = readFrame(videoReader);
        if ismember(frameIndex, selectedFrames)
            writeVideo(videoWriter, frame);
        end
        frameIndex = frameIndex + 1;
    end
    
    % Close video writer
    close(videoWriter);
    fprintf('Resampled video %s to %d frames.\n', videoName, targetFrameCount);
end

fprintf('All videos have been resampled and saved to %s.\n', outputFolder);
