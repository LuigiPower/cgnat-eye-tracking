clear all; close all; clc;

%% Load Video file
videoFileReader = vision.VideoFileReader('mds_project_cose.mov');

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


points = [leftEyePupil; leftIris; rightEyePupil; rightIris];
oldPoints = points;

initialize(pointTracker, points, videoFrame);

videoPlayer  = vision.VideoPlayer('Position',...
    [100 100 [size(videoFrame, 2), size(videoFrame, 1)]+30]);

% Convert the first box into a list of 4 points
% This is needed to be able to visualize the rotation of the object.
bboxPointsLeft = bbox2points(leftEye);
bboxPointsRight = bbox2points(rightEye);
bboxLeftEye = SupportFunctions.points2bbox(bboxPointsLeft);
bboxRightEye = SupportFunctions.points2bbox(bboxPointsRight);
pointThreshold = size(points, 1)/2; % Points to lose before recovery
%pointThreshold = (10*2 + 2)/2; % Points to lose before recovery
retryCount = 0;
retryMax = 5;

while ~isDone(videoFileReader)
    % get the next frame
    videoFrame = step(videoFileReader);
    
    %faceDetector = vision.CascadeObjectDetector();
    %visiblePoints = detectMinEigenFeatures(rgb2gray(videoFrame), 'ROI', bbox(2, :));

    % Track the points. Note that some points may be lost.
    [points, isFound] = step(pointTracker, videoFrame);
    visiblePoints = points(isFound, :);
    oldInliers = oldPoints(isFound, :);
    
    if size(visiblePoints, 1) >= pointThreshold % need at least 2 points
        retryCount = 0;
        
        % Estimate the geometric transformation between the old points
        % and the new points and eliminate outliers
        [xform, oldInliers, visiblePoints] = estimateGeometricTransform(...
            oldInliers, visiblePoints, 'similarity', 'MaxDistance', 4);
        
        % Apply the transformation to the bounding box points
        bboxPointsLeft = transformPointsForward(xform, double(bboxPointsLeft));
        bboxPointsRight = transformPointsForward(xform, double(bboxPointsRight));
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
        oldPoints = visiblePoints;
        setPoints(pointTracker, oldPoints);
    elseif retryCount < retryMax
        %%Recover points knowing last bounding box
        % (try this for a few frames, then try and recover the whole face)
        [leftEye, rightEye] = DetectionHelper.recoverPoints(videoFrame, bboxLeftEye, bboxRightEye, clusters);
        detected = [leftEyePupil; leftIris; rightEyePupil; rightIris];
        if size(detected, 1) > 0
            oldPoints = detected;
            setPoints(pointTracker, oldPoints);
        end
        retryCount = retryCount + 1;
    else
        %%Recover points knowing nothing
        [leftEye, rightEye, leftEyePupil, leftIris, rightEyePupil, rightIris] = DetectionHelper.recoverPointsFromScratch(videoFrame, clusters);
        detected = [leftEyePupil; leftIris; rightEyePupil; rightIris];
        if size(detected, 1) > 0
            oldPoints = detected;
            setPoints(pointTracker, oldPoints);

            bboxPointsLeft = bbox2points(leftEye);
            bboxPointsRight = bbox2points(rightEye);
            bboxLeftEye = SupportFunctions.points2bbox(bboxPointsLeft);
            bboxRightEye = SupportFunctions.points2bbox(bboxPointsRight);
        end
    end
    
    videoFrame = insertObjectAnnotation(videoFrame, 'Rectangle', bboxLeftEye, 'Left Eye');
    videoFrame = insertObjectAnnotation(videoFrame, 'Rectangle', bboxRightEye, 'Right Eye');
    
    % Display the annotated video frame using the video player object
    step(videoPlayer, videoFrame);
    
    %w = waitforbuttonpress;
    %if w == 0
    %    disp('Button click')
    %else
    %    disp('Key press')
    %end
end

