clear; close all; clc;

faceDetector = vision.CascadeObjectDetector();

%% Load Video file
videoFileReader = vision.VideoFileReader('mds_project_xxx.mov');
videoFrame      = step(videoFileReader);
bbox            = faceDetector(videoFrame);

% Draw the returned bounding box around the detected face.
videoFrame = insertShape(videoFrame, 'Rectangle', bbox);
figure; imshow(videoFrame); title('Detected face');

