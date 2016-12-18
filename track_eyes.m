clear all; close all; clc;

%% Load Video file
videoFileReader = vision.VideoFileReader('mds_project_cose.mov');
videoFrame      = step(videoFileReader);
videoFrame      = step(videoFileReader);
videoFrame      = step(videoFileReader);
videoFrame      = step(videoFileReader);
% TODO if we do this on frame 4, nothing works (left eye is wrooong)

faceDetector = vision.CascadeObjectDetector();
eyeDetector = vision.CascadeObjectDetector('EyePairSmall', 'UseROI', true);
eyeBigDetector = vision.CascadeObjectDetector('EyePairBig', 'UseROI', true);
leftEyeDetector = vision.CascadeObjectDetector('LeftEye', 'UseROI', true);
rightEyeDetector = vision.CascadeObjectDetector('RightEye', 'UseROI', true);
noseDetector = vision.CascadeObjectDetector('Nose', 'UseROI', true);

bbox = faceDetector(videoFrame);
[maximum, indexes] = SupportFunctions.orderDescByArea(bbox);

face_of_interest = bbox(indexes(1), :);
eyes = eyeDetector(videoFrame, face_of_interest);
eyePairBig = eyeBigDetector(videoFrame, face_of_interest);
nose = noseDetector(videoFrame, face_of_interest);

leftEyes = leftEyeDetector(videoFrame, face_of_interest);
rightEyes = rightEyeDetector(videoFrame, face_of_interest);

%% Need some processing to find the correct Left Eye and Right Eye
% by using the "eyes" Bounding Box, and then picking the best box
[eyesnum, ~] = size(eyes);
threshold = 0.1;
if(eyesnum > 0)
    leftEyes = SupportFunctions.removeNonIntersecting(leftEyes, eyes, threshold);
    rightEyes = SupportFunctions.removeNonIntersecting(rightEyes, eyes, threshold);
end

leftEye = SupportFunctions.getRightMost(leftEyes);
rightEye = SupportFunctions.getLeftMost(rightEyes);

%% Eye finding
[leftEyePupil, leftIris] = DetectionHelper.findEye(videoFrame, leftEye, 6);
[rightEyePupil, rightIris] = DetectionHelper.findEye(videoFrame, rightEye, 6);

%% Draw the returned bounding box around the detected face.
videoFrameShow = insertObjectAnnotation(videoFrame, 'Rectangle', face_of_interest, 'Face');
videoFrameShow = insertObjectAnnotation(videoFrameShow, 'Rectangle', eyes, 'Eyes');
videoFrameShow = insertObjectAnnotation(videoFrameShow, 'Rectangle', eyePairBig, 'EyePairBig');
videoFrameShow = insertObjectAnnotation(videoFrameShow, 'Rectangle', leftEye, 'Left Eye');
videoFrameShow = insertObjectAnnotation(videoFrameShow, 'Rectangle', rightEye, 'Right Eye');
videoFrameShow = insertObjectAnnotation(videoFrameShow, 'Rectangle', nose, 'Nose');

videoFrameShow = insertMarker(videoFrameShow, leftEyePupil, '+', 'Color', 'red');
videoFrameShow = insertMarker(videoFrameShow, rightEyePupil, '+', 'Color', 'red');

for i = 1:size(leftIris, 1)
    videoFrameShow = insertMarker(videoFrameShow, leftIris(i, :), '+', 'Color', 'green');
    videoFrameShow = insertMarker(videoFrameShow, rightIris(i, :), '+', 'Color', 'green');
end
% Show The first frame, with all found features
figure; imshow(videoFrameShow); title('Detected face');

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

while ~isDone(videoFileReader)
    % get the next frame
    videoFrame = step(videoFileReader);
    
    %faceDetector = vision.CascadeObjectDetector();
    %visiblePoints = detectMinEigenFeatures(rgb2gray(videoFrame), 'ROI', bbox(2, :));

    % Track the points. Note that some points may be lost.
    %[points, isFound] = step(pointTracker, videoFrame);
    %visiblePoints = points(isFound, :);
    %oldInliers = oldPoints(isFound, :);
    
    %if size(visiblePoints, 1) >= 2 % need at least 2 points
        
        % Estimate the geometric transformation between the old points
        % and the new points and eliminate outliers
        %[xform, oldInliers, visiblePoints] = estimateGeometricTransform(...
        %    oldInliers, visiblePoints, 'similarity', 'MaxDistance', 4);
        
        % Apply the transformation to the bounding box points
        %bboxPointsLeft = transformPointsForward(xform, bboxPointsLeft);
        %bboxPointsRight = transformPointsForward(xform, bboxPointsRight);
        
        % Insert a bounding box around the object being tracked
        %bboxPolygonLeft = reshape(bboxPointsLeft', 1, []);
        %bboxPolygonRight = reshape(bboxPointsRight', 1, []);
        %videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygonLeft, ...
        %    'LineWidth', 2);
        %videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygonRight, ...
        %    'LineWidth', 2);
        

        bbox = faceDetector(videoFrame);
        [maximum, indexes] = SupportFunctions.orderDescByArea(bbox);

        face_of_interest = bbox(indexes(1), :);
        eyes = eyeDetector(videoFrame, face_of_interest);

        leftEyes = leftEyeDetector(videoFrame, face_of_interest);
        rightEyes = rightEyeDetector(videoFrame, face_of_interest);

        %% Need some processing to find the correct Left Eye and Right Eye
        % by using the "eyes" Bounding Box, and then picking the best box
        threshold = 0.1;
        leftEyes = SupportFunctions.removeNonIntersecting(leftEyes, eyes, threshold);
        rightEyes = SupportFunctions.removeNonIntersecting(rightEyes, eyes, threshold);

        leftEye = SupportFunctions.getRightMost(leftEyes);
        rightEye = SupportFunctions.getLeftMost(rightEyes);
        
        %% Eye finding
        [leftEyePupil, leftIris] = DetectionHelper.findEye(videoFrame, leftEye, 6);
        [rightEyePupil, rightIris] = DetectionHelper.findEye(videoFrame, rightEye, 6);
        points = [leftEyePupil; leftIris; rightEyePupil; rightIris];

        % Display tracked points
        videoFrame = insertMarker(videoFrame, points, '+', ...
            'Color', 'white');
        
        videoFrame = insertObjectAnnotation(videoFrame, 'Rectangle', face_of_interest, 'Face');
        videoFrame = insertObjectAnnotation(videoFrame, 'Rectangle', eyes, 'Eyes');
        videoFrame = insertObjectAnnotation(videoFrame, 'Rectangle', leftEyes, 'Left Eye');
        videoFrame = insertObjectAnnotation(videoFrame, 'Rectangle', rightEyes, 'Right Eye');
        videoFrame = insertObjectAnnotation(videoFrame, 'Rectangle', leftEye, 'LEFT', 'Color', 'green');
        videoFrame = insertObjectAnnotation(videoFrame, 'Rectangle', rightEye, 'RIGHT', 'Color', 'green');
        % Reset the points
        %oldPoints = visiblePoints;
        %setPoints(pointTracker, oldPoints);
    %end
    
    % Display the annotated video frame using the video player object
    step(videoPlayer, videoFrame);
    
    w = waitforbuttonpress;
    if w == 0
        disp('Button click')
    else
        disp('Key press')
    end
end

