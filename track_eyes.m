clear; close all; clc;

%% Load Video file
videoFileReader = vision.VideoFileReader('mds_project_xxx.mov');
videoFrame      = step(videoFileReader);

faceDetector = vision.CascadeObjectDetector();
eyeDetector = vision.CascadeObjectDetector('EyePairSmall', 'UseROI', true);
leftEyeDetector = vision.CascadeObjectDetector('LeftEye', 'UseROI', true);
rightEyeDetector = vision.CascadeObjectDetector('RightEye', 'UseROI', true);
noseDetector = vision.CascadeObjectDetector('Nose', 'UseROI', true);

bbox = faceDetector(videoFrame);
[maximum, indexes] = SupportFunctions.orderDescByArea(bbox);

face_of_interest = bbox(indexes(1), :);
eyes = eyeDetector(videoFrame, face_of_interest);
nose = noseDetector(videoFrame, face_of_interest);

leftEyes = leftEyeDetector(videoFrame, face_of_interest);
rightEyes = rightEyeDetector(videoFrame, face_of_interest);

%% Need some processing to find the correct Left Eye and Right Eye
% by using the "eyes" Bounding Box, and then picking the best box
leftEyes = SupportFunctions.removeNonIntersecting(leftEyes, eyes);
rightEyes = SupportFunctions.removeNonIntersecting(rightEyes, eyes);

leftEye = SupportFunctions.getRightMost(leftEyes);
rightEye = SupportFunctions.getLeftMost(rightEyes);

%% Get Points to track in the video
leftEyePupil = SupportFunctions.getCenter(leftEye);
rightEyePupil = SupportFunctions.getCenter(rightEye);

%% Draw the returned bounding box around the detected face.
videoFrame = insertObjectAnnotation(videoFrame, 'Rectangle', face_of_interest, 'Face');
videoFrame = insertObjectAnnotation(videoFrame, 'Rectangle', eyes, 'Eyes');
videoFrame = insertObjectAnnotation(videoFrame, 'Rectangle', leftEye, 'Left Eye');
videoFrame = insertObjectAnnotation(videoFrame, 'Rectangle', rightEye, 'Right Eye');
videoFrame = insertObjectAnnotation(videoFrame, 'Rectangle', nose, 'Nose');

videoFrame = insertMarker(videoFrame, leftEyePupil, '+', 'Color', 'red');
videoFrame = insertMarker(videoFrame, rightEyePupil, '+', 'Color', 'red');

% Show The first frame, with all found features
figure; imshow(videoFrame); title('Detected face');

