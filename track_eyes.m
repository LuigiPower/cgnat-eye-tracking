clear all; close all; clc;

%% Load Video file
filename = 'mds_project_mad.mov';
videoFileReader = vision.VideoFileReader(filename);
videoForFrameCount = VideoReader(filename);
lastFrame = read(videoForFrameCount, inf);
totalFrameNumber = videoForFrameCount.NumberOfFrames;

% skip 1: clear iris
% skip 40: half iris
% skip 60: iris on topleft corner
skipFrames = 1;

for i = 1:skipFrames
    videoFrame      = step(videoFileReader);
end

clusters = 6;
[leftEye, rightEye, leftEyePupil, leftIris, rightEyePupil, rightIris] = DetectionHelper.recoverPointsFromScratch(videoFrame, clusters);

%% Draw the returned bounding box around the detected face.
% videoFrameShow = insertObjectAnnotation(videoFrame, 'Rectangle', face_of_interest, 'Face');
% videoFrameShow = insertObjectAnnotation(videoFrameShow, 'Rectangle', eyes, 'Eyes');
% videoFrameShow = insertObjectAnnotation(videoFrameShow, 'Rectangle', eyePairBig, 'EyePairBig');
% videoFrameShow = insertObjectAnnotation(videoFrameShow, 'Rectangle', leftEye, 'Left Eye');
% videoFrameShow = insertObjectAnnotation(videoFrameShow, 'Rectangle', rightEye, 'Right Eye');
% videoFrameShow = insertObjectAnnotation(videoFrameShow, 'Rectangle', nose, 'Nose');

% videoFrameShow = insertMarker(videoFrameShow, leftEyePupil, '+', 'Color', 'red');
% videoFrameShow = insertMarker(videoFrameShow, rightEyePupil, '+', 'Color', 'red');

% for i = 1:size(leftIris, 1)
%     videoFrameShow = insertMarker(videoFrameShow, leftIris(i, :), '+', 'Color', 'green');
%     videoFrameShow = insertMarker(videoFrameShow, rightIris(i, :), '+', 'Color', 'green');
% end
% Show The first frame, with all found features
% figure; imshow(videoFrameShow); title('Detected face');

%% Video tracking
% Create a point tracker and enable the bidirectional error constraint to
% make it more robust in the presence of noise and clutter.
pointTracker = vision.PointTracker('MaxBidirectionalError', 2);

%points = [leftEyePupil; rightEyePupil];
greyscaleVideoFrame = rgb2gray(videoFrame);
%leftPoints = detectMinEigenFeatures(greyscaleVideoFrame, 'ROI', leftEye);
%rightPoints = detectMinEigenFeatures(greyscaleVideoFrame, 'ROI', rightEye);
%leftPoints = detectHarrisFeatures(rgb2gray(videoFrame), 'ROI', leftEye);
%rightPoints = detectHarrisFeatures(rgb2gray(videoFrame), 'ROI', rightEye);
%imshow(greyscaleVideoFrame);

%% Initialization
xLeftEye = zeros(totalFrameNumber, 1);
yLeftEye = zeros(totalFrameNumber, 1);
xRightEye = zeros(totalFrameNumber, 1);
yRightEye = zeros(totalFrameNumber, 1);
frameCount = 1;

% Convert the first box into a list of 4 points
% This is needed to be able to visualize the rotation of the object.
bboxPointsLeft = double(bbox2points(leftEye));
bboxPointsRight = double(bbox2points(rightEye));
bboxLeftEye = SupportFunctions.points2bbox(bboxPointsLeft);
bboxRightEye = SupportFunctions.points2bbox(bboxPointsRight);
retryLeftCount = 0;
retryRightCount = 0;
retryMax = 5;

points = [leftEyePupil; rightEyePupil; bboxPointsLeft; bboxPointsRight];
oldPoints = points;
initialize(pointTracker, points, videoFrame);
%pointThreshold = (10*2 + 2)/2; % Points to lose before recovery
pointThreshold = size(points, 1)/2; % Points to lose before recovery

%% Play the video and track both eyes
videoPlayer  = vision.VideoPlayer('Position',...
    [100 100 [size(videoFrame, 2), size(videoFrame, 1)]+30]);

while ~isDone(videoFileReader)
    % get the next frame
    videoFrame = step(videoFileReader);
    
    %faceDetector = vision.CascadeObjectDetector();
    %visiblePoints = detectMinEigenFeatures(rgb2gray(videoFrame), 'ROI', bbox(2, :));

    % Track the points. Note that some points may be lost.
    [points, isFound] = step(pointTracker, videoFrame);
    
    %% Recovering lost points
    if isFound(1) == 0 % LeftEyePupil lost tracking
        if retryLeftCount < retryMax
            %%Recover points knowing last bounding box
            % (try this for a few frames, then try and recover the whole face)
            [leftEyePupil, ~, ~, ~] = DetectionHelper.recoverPoints(videoFrame, bboxLeftEye, [], clusters);
            if size(leftEyePupil, 1) > 0
                points(1, :) = leftEyePupil;
            end
            retryLeftCount = retryLeftCount + 1;
        else
            %%Recover points knowing nothing
            [leftEye, ~, leftEyePupil, ~, ~, ~] = DetectionHelper.recoverPointsFromScratch(videoFrame, clusters, 1);
            if size(leftEye, 1) > 0
                bboxPointsLeft = double(bbox2points(leftEye));
                bboxLeftEye = SupportFunctions.points2bbox(bboxPointsLeft);
                
                % set bboxPointsLeft to points(3, 4, 5, 6)
                points(1, :) = leftEyePupil;
                points(3:6, :) = bboxPointsLeft;
            end
        end
    else
        retryLeftCount = 0;
    end
    
    if isFound(2) == 0 %RightEyePupil lost tracking
        if retryRightCount < retryMax
            %%Recover points knowing last bounding box
            % (try this for a few frames, then try and recover the whole face)
            [~, ~, rightEyePupil, ~] = DetectionHelper.recoverPoints(videoFrame, [], bboxRightEye, clusters);
            if size(rightEyePupil, 1) > 0
                points(2, :) = rightEyePupil;
            end
            retryRightCount = retryRightCount + 1;
        else
            %%Recover points knowing nothing
            [~, rightEye, ~, ~, rightEyePupil, ~] = DetectionHelper.recoverPointsFromScratch(videoFrame, clusters, 2);
            if size(detected, 1) > 0
                bboxPointsRight = double(bbox2points(rightEye));
                bboxRightEye = SupportFunctions.points2bbox(bboxPointsRight);

                % set bboxPointsRight to points(7, 8, 9, 10)
                points(1, :) = rightEyePupil;
                points(7:10, :) = bboxPointsRight;
            end
        end
    else
        retryRightCount = 0;
    end
    
    % TODO same stuff for the bounding boxes
    
    %% Calculating X and Y movement of pupils
    leftBoxCenter = mean(points(3:6, :));
    rightBoxCenter = mean(points(7:10, :));
    xLeftEye(frameCount) = leftBoxCenter(1) - points(1, 1);
    yLeftEye(frameCount) = leftBoxCenter(2) - points(1, 2);
    xRightEye(frameCount) = rightBoxCenter(1) - points(2, 1);
    yRightEye(frameCount) = rightBoxCenter(2) - points(2, 2);
    frameCount = frameCount + 1;
    %% Showing the points
    %visiblePoints = points(isFound, :);
    visiblePoints = points;
    oldInliers = oldPoints;
    
    %if size(visiblePoints, 1) >= pointThreshold % need at least 2 points
    % Estimate the geometric transformation between the old points
    % and the new points and eliminate outliers
    [xform, oldInliers, visiblePoints] = estimateGeometricTransform(...
        oldInliers, visiblePoints, 'similarity', 'MaxDistance', 4);

    % Apply the transformation to the bounding box points
    bboxPointsLeft = transformPointsForward(xform, bboxPointsLeft);
    bboxPointsRight = transformPointsForward(xform, bboxPointsRight);
    bboxLeftEye = SupportFunctions.points2bbox(bboxPointsLeft);
    bboxRightEye = SupportFunctions.points2bbox(bboxPointsRight);

    % Insert a bounding box around the object being tracked
    bboxPolygonLeft = reshape(bboxPointsLeft', 1, []);
    bboxPolygonRight = reshape(bboxPointsRight', 1, []);
    videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygonLeft, ...
        'LineWidth', 2);
    videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygonRight, ...
        'LineWidth', 2);

    % Display tracked points
    videoFrame = insertMarker(videoFrame, points, '+', ...
        'Color', 'white');

    % Reset the points
    oldPoints = points;
    setPoints(pointTracker, oldPoints);
    %end
    
    videoFrame = insertObjectAnnotation(videoFrame, 'Rectangle', bboxLeftEye, 'Left Eye');
    videoFrame = insertObjectAnnotation(videoFrame, 'Rectangle', bboxRightEye, 'Right Eye');
    
    % Display the annotated video frame using the video player object
    step(videoPlayer, videoFrame);
end

%% Plot X and Y position of both eyes
x = 1:totalFrameNumber;
figure; plot(x, xLeftEye, x, yLeftEye); title('Left Eye'); xlabel('Frame'); ylabel('Distance in Pixels');
figure; plot(x, xRightEye, x, yRightEye); title('Right Eye'); xlabel('Frame'); ylabel('Distance in Pixels');