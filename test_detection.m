clear all; close all; clc;

%% Load Video file
%filename = 'uncharted4first.mp4';
%filename = 'uncharted4second.mp4';
filename = 'mds_project_cose.mov';
%filename = 'mds_project_xxx.mov';
%filename = 'mds_project_mad.mov';
%filename = 'mds_project_hard.mov';
%filename = 'mds_project_ooo.mov';
%filename = 'mds_project.mov';

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

faceDetector = vision.CascadeObjectDetector();
eyeDetector = vision.CascadeObjectDetector('EyePairSmall', 'UseROI', true);
leftEyeDetector = vision.CascadeObjectDetector('LeftEye', 'UseROI', true);
rightEyeDetector = vision.CascadeObjectDetector('RightEye', 'UseROI', true);
eyeBigDetector = vision.CascadeObjectDetector('EyePairBig', 'UseROI', true);
noseDetector = vision.CascadeObjectDetector('Nose', 'UseROI', true);

%% Testing it
%videoFrame = rgb2gray(videoFrame);
%videoFrame = imadjust(videoFrame);

bbox = faceDetector(videoFrame);
[~, indexes] = SupportFunctions.orderDescByArea(bbox);

if size(indexes, 1) == 0
    return;
end

face_of_interest = bbox(indexes(1), :);
eyes = eyeBigDetector(videoFrame, face_of_interest);
if size(eyes, 1) == 0
    eyes = eyeDetector(videoFrame, face_of_interest);
end

[~, indexes] = SupportFunctions.orderDescByArea(eyes);

if size(indexes, 1) == 0
    return;
end

threshold = 0.3;

leftEyes = leftEyeDetector(videoFrame, eyes(indexes(1), :));
rightEyes = rightEyeDetector(videoFrame, eyes(indexes(1), :));
[~, leftIndexes] = SupportFunctions.orderDescByArea(leftEyes);
[~, rightIndexes] = SupportFunctions.orderDescByArea(rightEyes);
totalEyes = [leftEyes; rightEyes];
%totalEyes = SupportFunctions.removeNonIntersecting(totalEyes, eyes, threshold);

nose = noseDetector(videoFrame, face_of_interest);

%% Draw the returned bounding box around the detected face.
videoFrameShow = insertObjectAnnotation(videoFrame, 'Rectangle', face_of_interest, 'Face', 'Color', {'yellow'});
videoFrameShow = insertObjectAnnotation(videoFrameShow, 'Rectangle', eyes, 'Eyes', 'Color', {'blue'});
% videoFrameShow = insertObjectAnnotation(videoFrameShow, 'Rectangle', eyePairBig, 'EyePairBig');
videoFrameShow = insertObjectAnnotation(videoFrameShow, 'Rectangle', leftEyes, 'Left Eye', 'Color', {'green'});
videoFrameShow = insertObjectAnnotation(videoFrameShow, 'Rectangle', rightEyes, 'Right Eye', 'Color', {'white'});
videoFrameShow = insertObjectAnnotation(videoFrameShow, 'Rectangle', nose, 'Nose', 'Color', {'black'});

%videoFrameShow = insertMarker(videoFrameShow, leftEyePupil, '+', 'Color', 'red');
%videoFrameShow = insertMarker(videoFrameShow, rightEyePupil, '+', 'Color', 'red');

% for i = 1:size(leftIris, 1)
%     videoFrameShow = insertMarker(videoFrameShow, leftIris(i, :), '+', 'Color', 'green');
%     videoFrameShow = insertMarker(videoFrameShow, rightIris(i, :), '+', 'Color', 'green');
% end

% Show The first frame, with all found features
figure; imshow(videoFrameShow); title('Detected face');

