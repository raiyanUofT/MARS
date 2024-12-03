% Script: segment_videos_2d.m
% Segment 2D videos into overlapping segments.

% Initialize variables
inputFolder = './videos_2d'; % Input folder with 2D videos
segmentFolder = './segments_2d'; % Output folder for segmented videos
overlapFraction = 0.2; % 20% overlap
frameRate = 10; % Frame rate for the segments

% Create output directory if not exist
if ~exist(segmentFolder, 'dir')
    mkdir(segmentFolder);
end

% Get list of video files in the input folder
videoFiles = dir(fullfile(inputFolder, '*.mp4'));
fprintf('Found %d videos to segment.\n', length(videoFiles));

% Process each video
fprintf('Segmenting videos...\n');
for i = 1:length(videoFiles)
    % Load video file
    videoPath = fullfile(inputFolder, videoFiles(i).name);
    videoReader = VideoReader(videoPath);
    totalFrames = videoReader.NumFrames;
    framesPerSegment = ceil(totalFrames / 2); % Half the frames in each segment
    
    % Calculate overlap
    overlapFrames = ceil(overlapFraction * framesPerSegment);
    segment1Start = 1;
    segment1End = segment1Start + framesPerSegment - 1;
    segment2Start = segment1End - overlapFrames + 1;
    segment2End = totalFrames;

    % Define segment file names
    [~, videoName, ~] = fileparts(videoFiles(i).name);
    segment1Name = sprintf('%s_segment1.mp4', videoName);
    segment2Name = sprintf('%s_segment2.mp4', videoName);
    
    segment1Path = fullfile(segmentFolder, segment1Name);
    segment2Path = fullfile(segmentFolder, segment2Name);
    
    % Write first segment
    videoWriter1 = VideoWriter(segment1Path, 'MPEG-4');
    videoWriter1.FrameRate = frameRate;
    open(videoWriter1);
    for f = segment1Start:segment1End
        if hasFrame(videoReader)
            frame = read(videoReader, f);
            writeVideo(videoWriter1, frame);
        end
    end
    close(videoWriter1);
    
    % Write second segment
    videoWriter2 = VideoWriter(segment2Path, 'MPEG-4');
    videoWriter2.FrameRate = frameRate;
    open(videoWriter2);
    for f = segment2Start:segment2End
        if hasFrame(videoReader)
            frame = read(videoReader, f);
            writeVideo(videoWriter2, frame);
        end
    end
    close(videoWriter2);
    
    fprintf('Segmented video: %s into two segments.\n', videoFiles(i).name);
end

fprintf('Segmented videos saved in %s.\n', segmentFolder);
