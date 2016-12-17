clear; close all; clc;

%% Load Video file
videoFileReader = vision.VideoFileReader('mds_project_mad.mov');
videoFrame      = step(videoFileReader);

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
if(eyesnum > 0)
    leftEyes = SupportFunctions.removeNonIntersecting(leftEyes, eyes);
    rightEyes = SupportFunctions.removeNonIntersecting(rightEyes, eyes);
end

leftEye = SupportFunctions.getRightMost(leftEyes);
rightEye = SupportFunctions.getLeftMost(rightEyes);

%% Get Points to track in the video
leftEyePupil = SupportFunctions.getCenter(leftEye);
rightEyePupil = SupportFunctions.getCenter(rightEye);

leftEyeImage = imcrop(videoFrame, leftEye);

%videoFrame(leftEye(1, 2):(leftEye(1, 2) + leftEye(1, 4)), ...
%            leftEye(1, 1):(leftEye(1, 1) + leftEye(1, 3))), ...

%% Left Eye analysis
[w, h] = size(leftEyeImage);
%[centers, radii, metric] = imfindcircles(leftEyeImage, h/3);

%[r, ~] = size(centers);
%for i = 1:r
%    leftEyeImage = insertObjectAnnotation(leftEyeImage, 'Circle', [centers(i, :) radii(i)], 'circle');
%end
leftEyeImage = histeq(rgb2gray(leftEyeImage));
figure, imshow(leftEyeImage);
mask = zeros(size(leftEyeImage));
mask(25:end-25,25:end-25) = 1;

figure, imshow(mask);
title('Initial Contour Location');

bw = activecontour(leftEyeImage, mask, 300);

figure, imshow(bw);
title('Segmented Image');

%{
%% Clustering
cform = makecform('srgb2lab');

leftEyeImageGS = rgb2gray(leftEyeImage);
leftEyeImage(:, :, 1) = leftEyeImageGS;
leftEyeImage(:, :, 2) = leftEyeImageGS;
leftEyeImage(:, :, 3) = leftEyeImageGS;
%imshow(leftEyeImage);

lab_leftEyeImage = applycform(double(leftEyeImage), cform);

ab = double(lab_leftEyeImage(:, :, 2:3));
nrows = size(ab, 1);
ncols = size(ab, 2);
ab = reshape(ab, nrows * ncols, 2);

nColors = 2;
% repeat the clustering 3 times to avoid local minima
[cluster_idx, cluster_center] = kmeans(ab, nColors, 'distance', 'sqEuclidean', ...
                                      'Replicates', 3);
                                  
pixel_labels = reshape(cluster_idx, nrows, ncols);

imshow(pixel_labels, []), title('image labeled by cluster index');
%}

%% Draw the returned bounding box around the detected face.
videoFrameShow = insertObjectAnnotation(videoFrame, 'Rectangle', face_of_interest, 'Face');
videoFrameShow = insertObjectAnnotation(videoFrameShow, 'Rectangle', eyes, 'Eyes');
videoFrameShow = insertObjectAnnotation(videoFrameShow, 'Rectangle', eyePairBig, 'EyePairBig');
videoFrameShow = insertObjectAnnotation(videoFrameShow, 'Rectangle', leftEye, 'Left Eye');
videoFrameShow = insertObjectAnnotation(videoFrameShow, 'Rectangle', rightEye, 'Right Eye');
videoFrameShow = insertObjectAnnotation(videoFrameShow, 'Rectangle', nose, 'Nose');

%[r, ~] = size(centers);
%for i = 1:r
%    videoFrameShow = insertObjectAnnotation(videoFrameShow, 'Circle', [(centers(i) + leftEye(1, 1:2)) radii(i)], 'circle');
%end

videoFrameShow = insertMarker(videoFrameShow, leftEyePupil, '+', 'Color', 'red');
videoFrameShow = insertMarker(videoFrameShow, rightEyePupil, '+', 'Color', 'red');

% Show The first frame, with all found features
%figure; imshow(videoFrameShow); title('Detected face');

%% Video tracking
% Create a point tracker and enable the bidirectional error constraint to
% make it more robust in the presence of noise and clutter.
pointTracker = vision.PointTracker('MaxBidirectionalError', 2);

%points = [leftEyePupil; rightEyePupil];
greyscaleVideoFrame = rgb2gray(videoFrame);
leftPoints = detectMinEigenFeatures(greyscaleVideoFrame, 'ROI', leftEye);
rightPoints = detectMinEigenFeatures(greyscaleVideoFrame, 'ROI', rightEye);
%leftPoints = detectHarrisFeatures(rgb2gray(videoFrame), 'ROI', leftEye);
%rightPoints = detectHarrisFeatures(rgb2gray(videoFrame), 'ROI', rightEye);
%imshow(greyscaleVideoFrame);

points = [leftPoints.Location; rightPoints.Location];
oldPoints = points;

initialize(pointTracker, points, videoFrame);

videoPlayer  = vision.VideoPlayer('Position',...
    [100 100 [size(videoFrame, 2), size(videoFrame, 1)]+30]);

% Convert the first box into a list of 4 points
% This is needed to be able to visualize the rotation of the object.
bboxPointsLeft = bbox2points(leftEye);
bboxPointsRight = bbox2points(rightEye);
%{
while ~isDone(videoFileReader)
    % get the next frame
    videoFrame = step(videoFileReader);
    
    %faceDetector = vision.CascadeObjectDetector();
    %visiblePoints = detectMinEigenFeatures(rgb2gray(videoFrame), 'ROI', bbox(2, :));

    % Track the points. Note that some points may be lost.
    [points, isFound] = step(pointTracker, videoFrame);
    visiblePoints = points(isFound, :);
    oldInliers = oldPoints(isFound, :);
    
    if size(visiblePoints, 1) >= 2 % need at least 2 points
        
        % Estimate the geometric transformation between the old points
        % and the new points and eliminate outliers
        [xform, oldInliers, visiblePoints] = estimateGeometricTransform(...
            oldInliers, visiblePoints, 'similarity', 'MaxDistance', 4);
        
        % Apply the transformation to the bounding box points
        bboxPointsLeft = transformPointsForward(xform, bboxPointsLeft);
        bboxPointsRight = transformPointsForward(xform, bboxPointsRight);
        
        % Insert a bounding box around the object being tracked
        bboxPolygonLeft = reshape(bboxPointsLeft', 1, []);
        bboxPolygonRight = reshape(bboxPointsRight', 1, []);
        videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygonLeft, ...
            'LineWidth', 2);
        videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygonRight, ...
            'LineWidth', 2);
                
        % Display tracked points
        videoFrame = insertMarker(videoFrame, visiblePoints, '+', ...
            'Color', 'white');
        
        % Reset the points
        oldPoints = visiblePoints;
        setPoints(pointTracker, oldPoints);
    end
    
    % Display the annotated video frame using the video player object
    step(videoPlayer, videoFrame);
end
%}
