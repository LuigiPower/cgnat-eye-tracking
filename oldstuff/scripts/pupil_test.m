clear all; close all; clc;

%% Load Video file
%filename = 'uncharted4first.mp4';
filename = 'uncharted4second.mp4';
% filename = 'mds_project_cose.mov';
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

%% Play the video and track both eyes
videoPlayer  = vision.VideoPlayer('Position',...
    [100 100 [size(videoFrame, 2), size(videoFrame, 1)]+30]);


while ~isDone(videoFileReader)
    videoFrame = step(videoFileReader);
    [leftEye, rightEye, leftEyePupil, leftIris, rightEyePupil, rightIris] = PupilTestHelper.recoverPointsFromScratch(videoFrame);
%     figure; imshow(videoFrame); title('Detected face');
    leftEye
    rightEye
    leftEyePupil
    rightEyePupil
%     leftIris
%     rightIris
    videoFrameShow = videoFrame;
    %% Draw the returned bounding box around the detected face.
    % videoFrameShow = insertObjectAnnotation(videoFrame, 'Rectangle', face_of_interest, 'Face');
    % videoFrameShow = insertObjectAnnotation(videoFrameShow, 'Rectangle', eyes, 'Eyes');
    % videoFrameShow = insertObjectAnnotation(videoFrameShow, 'Rectangle', eyePairBig, 'EyePairBig');
%     videoFrameShow = insertObjectAnnotation(videoFrameShow, 'Rectangle', leftEye, 'Left Eye');
%     videoFrameShow = insertObjectAnnotation(videoFrameShow, 'Rectangle', rightEye, 'Right Eye');
    % videoFrameShow = insertObjectAnnotation(videoFrameShow, 'Rectangle', nose, 'Nose');
% 
%     videoFrameShow = insertMarker(videoFrameShow, leftEyePupil, '+', 'Color', 'red');
%     videoFrameShow = insertMarker(videoFrameShow, rightEyePupil, '+', 'Color', 'red');
% 
%     for i = 1:size(leftIris, 1)
%         videoFrameShow = insertMarker(videoFrameShow, leftIris(i, :), '+', 'Color', 'green');
%         videoFrameShow = insertMarker(videoFrameShow, rightIris(i, :), '+', 'Color', 'green');
%     end
    % Show The first frame, with all found features
%     figure; imshow(videoFrameShow); title('Detected face');
    
     videoFrame = insertObjectAnnotation(videoFrame, 'Rectangle', leftEye, 'Left Eye');
     videoFrame = insertObjectAnnotation(videoFrame, 'Rectangle', rightEye, 'Right Eye');
     if size(leftEyePupil,1) > 0
              videoFrame = insertMarker(videoFrame, leftEyePupil, '+', 'Color', 'red');
     end
     if size(rightEyePupil,1) > 0

         videoFrame = insertMarker(videoFrame, rightEyePupil, '+', 'Color', 'red');
     end
     
%      if size(leftIris,1) > 0
%        for i = 1:size(leftIris, 1)
%             videoFrame = insertMarker(videoFrame, leftIris(i, :), '+', 'Color', 'green');
%             videoFrame = insertMarker(videoFrame, rightIris(i, :), '+', 'Color', 'green');
%         end   
%      end
    
    
%      w = waitforbuttonpress;
    step(videoPlayer, videoFrame);
end
