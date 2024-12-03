% run_2d_processing_scripts.m
% This script executes three MATLAB scripts in order to process 2D videos:
% 1. generate_2d_videos.m
% 2. segment_2d_videos.m
% 3. downsample_2d_videos.m

fprintf('Starting execution of 2D processing scripts...\n');

try
    % Step 1: Run the script to generate 2D videos
    fprintf('Running generate_2d_videos.m...\n');
    run('generate_2d_videos.m');
    fprintf('Completed generate_2d_videos.m\n');
    
    % Step 2: Run the script to segment 2D videos
    fprintf('Running segment_2d_videos.m...\n');
    run('segment_2d_videos.m');
    fprintf('Completed segment_2d_videos.m\n');
    
    % Step 3: Run the script to downsample 2D videos
    fprintf('Running downsample_2d_videos.m...\n');
    run('downsample_2d_videos.m');
    fprintf('Completed downsample_2d_videos.m\n');
    
    fprintf('All 2D processing scripts executed successfully.\n');

catch ME
    % Error handling
    fprintf('An error occurred while executing the scripts: %s\n', ME.message);
    if ~isempty(ME.stack)
        fprintf('Error occurred in script: %s at line %d\n', ME.stack(1).name, ME.stack(1).line);
    end
end
