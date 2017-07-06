clear all; close all; clc;

%% Load Video file
% filename = 'uncharted4first.mp4';
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
% Using Eye bounding boxes
xLeftEyeBox = zeros(totalFrameNumber, 1);
yLeftEyeBox = zeros(totalFrameNumber, 1);
xRightEyeBox = zeros(totalFrameNumber, 1);
yRightEyeBox = zeros(totalFrameNumber, 1);
% Using point between eyes
xLeftEyeCenter = zeros(totalFrameNumber, 1);
yLeftEyeCenter = zeros(totalFrameNumber, 1);
xRightEyeCenter = zeros(totalFrameNumber, 1);
yRightEyeCenter = zeros(totalFrameNumber, 1);
% TODO Try using nose
frameCount = 1;

% Convert the first box into a list of 4 points
% This is needed to be able to visualize the rotation of the object.
bboxPointsLeft = [1 1; 1 1; 1 1; 1 1];
bboxPointsRight = [1 1; 1 1; 1 1; 1 1];
if size(leftEye, 1) > 0
    bboxPointsLeft = double(bbox2points(leftEye));
    bboxLeftEye = SupportFunctions.points2bbox(bboxPointsLeft);
    leftEyeTracked = true;
    leftBoxTracked = true;
else
    leftEyeTracked = false;
    leftBoxTracked = false;
end

if size(rightEye, 1) > 0
    bboxPointsRight = double(bbox2points(rightEye));
    bboxRightEye = SupportFunctions.points2bbox(bboxPointsRight);
    rightEyeTracked = true;
    rightBoxTracked = true;
else
    rightEyeTracked = false;
    rightBoxTracked = false;
end

eyeCenter = mean([bboxPointsLeft;bboxPointsRight]);
referenceDistance = abs(mean(bboxPointsLeft) - mean(bboxPointsRight));
points = [leftEyePupil; rightEyePupil; bboxPointsLeft; bboxPointsRight; eyeCenter];
oldPoints = points;
initialize(pointTracker, points, videoFrame);

retryLeftCount = 0;
retryRightCount = 0;
retryMax = 30; % TODO A second, check the framerate
    
%% Play the video and track both eyes
videoPlayer  = vision.VideoPlayer('Position',...
    [100 100 [size(videoFrame, 2), size(videoFrame, 1)]+30]);

while ~isDone(videoFileReader)
    %% get the next frame
    videoFrame = step(videoFileReader);
    
    if leftEyeTracked && rightEyeTracked && size(leftEye, 1) > 0 && size(rightEye, 1) > 0
        [leftEyeTracked, rightEyeTracked] = DetectionHelper.checkOverlap(leftEye, rightEye);
    end
    
    %% Track the points. Note that some points may be lost.
    [points, isFound] = step(pointTracker, videoFrame);
    
    %% Recovering lost points
    if size(isFound, 1) ~= 11 || (~leftEyeTracked && ~rightEyeTracked) || (~leftBoxTracked && ~rightBoxTracked)
        [leftEye, rightEye, leftEyePupil, leftIris, rightEyePupil, rightIris] = DetectionHelper.recoverPointsFromScratch(videoFrame, clusters);
        disp('Recovering all the points');
        if size(leftEyePupil, 1) ~= 0 && size(rightEyePupil, 1) ~= 0 && size(leftEye, 1) ~= 0 && size(rightEye, 1) ~= 0
            bboxPointsLeft = double(bbox2points(leftEye));
            bboxPointsRight = double(bbox2points(rightEye));

            eyeCenter = mean([bboxPointsLeft;bboxPointsRight]);
            referenceDistance = abs(mean(bboxPointsLeft) - mean(bboxPointsRight));
            points = [leftEyePupil; rightEyePupil; bboxPointsLeft; bboxPointsRight; eyeCenter];
            oldPoints = points;
            
            leftEyeTracked = true;
            rightEyeTracked = true;
            leftBoxTracked = true;
            rightBoxTracked = true;
        end
    else
        if isFound(1) == 0 || ~leftEyeTracked % LeftEyePupil lost tracking
            leftEyeTracked = false;
            disp('Recovering Left Eye and Pupil using existing boxes');
            if retryLeftCount < retryMax
                %% Recover points knowing last bounding box
                % (try this for a few frames, then try and recover the whole face)
                [leftEyePupil, ~, ~, ~] = DetectionHelper.recoverPoints(videoFrame, bboxLeftEye, [], clusters);
                if size(leftEyePupil, 1) > 0
                    points(1, :) = leftEyePupil;
                    leftEyeTracked = true;
                end
                retryLeftCount = retryLeftCount + 1;
            else
                %% Recover points knowing nothing
                [leftEye, ~, leftEyePupil, ~, ~, ~] = DetectionHelper.recoverPointsFromScratch(videoFrame, clusters, 1);
                disp('Recovering Left Eye and Pupil from scratch');
                if size(leftEyePupil, 1) > 0
                    bboxPointsLeft = double(bbox2points(leftEye));

                    % set bboxPointsLeft to points(3, 4, 5, 6)
                    points(1, :) = leftEyePupil;
                    points(3:6, :) = bboxPointsLeft;
                    leftEyeTracked = true;
                    
                    eyeCenter = mean([bboxPointsLeft;bboxPointsRight]);
                    referenceDistance = abs(mean(bboxPointsLeft) - mean(bboxPointsRight));
                end
            end
        else
            retryLeftCount = 0;
        end

        if isFound(2) == 0 || ~rightEyeTracked % RightEyePupil lost tracking
            rightEyeTracked = false;
            disp('Recovering Right Eye and Pupil using existing boxes');
            if retryRightCount < retryMax
                %%Recover points knowing last bounding box
                % (try this for a few frames, then try and recover the whole face)
                [~, ~, rightEyePupil, ~] = DetectionHelper.recoverPoints(videoFrame, [], bboxRightEye, clusters);
                if size(rightEyePupil, 1) > 0
                    points(2, :) = rightEyePupil;
                    rightEyeTracked = true;
                end
                retryRightCount = retryRightCount + 1;
            else
                %%Recover points knowing nothing
                [~, rightEye, ~, ~, rightEyePupil, ~] = DetectionHelper.recoverPointsFromScratch(videoFrame, clusters, 2);
                disp('Recovering Right Eye and Pupil from scratch');
                if size(rightEyePupil, 1) > 0
                    bboxPointsRight = double(bbox2points(rightEye));

                    % set bboxPointsRight to points(7, 8, 9, 10)
                    points(2, :) = rightEyePupil;
                    points(7:10, :) = bboxPointsRight;
                    rightEyeTracked = true;
                    
                    eyeCenter = mean([bboxPointsLeft;bboxPointsRight]);
                    referenceDistance = abs(mean(bboxPointsLeft) - mean(bboxPointsRight));
                end
            end
        else
            retryRightCount = 0;
        end

        % same stuff for the bounding boxes
        if (prod(isFound(3:6)) == 0 || ~leftBoxTracked) && ~leftEyeTracked
            leftBoxTracked = false;
            [leftEye, ~, leftEyePupil, ~, ~, ~] = DetectionHelper.recoverPointsFromScratch(videoFrame, clusters, 1);
            disp('Recovering Bounding box Left');
            if size(leftEyePupil, 1) > 0
                bboxPointsLeft = double(bbox2points(leftEye));

                % set bboxPointsLeft to points(3, 4, 5, 6)
                points(1, :) = leftEyePupil;
                points(3:6, :) = bboxPointsLeft;
                
                eyeCenter = mean([bboxPointsLeft;bboxPointsRight]);
                referenceDistance = abs(mean(bboxPointsLeft) - mean(bboxPointsRight));
                leftBoxTracked = true;
            end
        end

        if (prod(isFound(7:10)) == 0 || ~rightBoxTracked) && ~rightEyeTracked
            rightBoxTracked = false;
            [~, rightEye, ~, ~, rightEyePupil, ~] = DetectionHelper.recoverPointsFromScratch(videoFrame, clusters, 2);
            disp('Recovering Bounding box Right');
            if size(rightEyePupil, 1) > 0
                bboxPointsRight = double(bbox2points(rightEye));

                % set bboxPointsRight to points(7, 8, 9, 10)
                points(2, :) = rightEyePupil;
                points(7:10, :) = bboxPointsRight;
                eyeCenter = mean([bboxPointsLeft;bboxPointsRight]);
                referenceDistance = abs(mean(bboxPointsLeft) - mean(bboxPointsRight));
                rightBoxTracked = true;
            end
        end
        
        %same stuff for the "center" point
        if isFound(11) == 0 && leftEyeTracked && rightEyeTracked
            disp('Recovering Center Point');
            eyeCenter = mean([bboxPointsLeft;bboxPointsRight]);
            referenceDistance = abs(mean(bboxPointsLeft) - mean(bboxPointsRight));
        end
        
        %% Calculating X and Y movement of pupils
        % Using Eye bounding boxes
        if leftEyeTracked
            leftBoxCenter = mean(points(3:6, :));
            xLeftEyeBox(frameCount) = leftBoxCenter(1) - points(1, 1);
            yLeftEyeBox(frameCount) = leftBoxCenter(2) - points(1, 2);
        end
        if rightEyeTracked
            rightBoxCenter = mean(points(7:10, :));
            xRightEyeBox(frameCount) = rightBoxCenter(1) - points(2, 1);
            yRightEyeBox(frameCount) = rightBoxCenter(2) - points(2, 2);
        end
        %Using Point between eyes
        pupilCenter = points(11, :);
        if leftEyeTracked
            xLeftEyeCenter(frameCount) = abs(pupilCenter(1) - points(1, 1));
            yLeftEyeCenter(frameCount) = abs(pupilCenter(2) - points(1, 2));
        end
        if rightEyeTracked
            xRightEyeCenter(frameCount) = abs(pupilCenter(1) - points(2, 1));
            yRightEyeCenter(frameCount) = abs(pupilCenter(2) - points(2, 2));
        end
    end
    
    frameCount = frameCount + 1;
    %% Showing the points
    %visiblePoints = points(isFound, :);
    visiblePoints = points;
    oldInliers = oldPoints;
    
    try
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
        
        videoFrame = insertObjectAnnotation(videoFrame, 'Rectangle', bboxLeftEye, 'Left Eye');
        videoFrame = insertObjectAnnotation(videoFrame, 'Rectangle', bboxRightEye, 'Right Eye');
    catch
        disp('Failed bboxes');
    end
    
    % Display tracked points
    videoFrame = insertMarker(videoFrame, points, '+', ...
        'Color', 'white');

    % Reset the points
    oldPoints = points;
    oldPoints(oldPoints < 0) = 1;
    setPoints(pointTracker, oldPoints);
    %end
    
    % Display the annotated video frame using the video player object
    step(videoPlayer, videoFrame);
end

%% Plot X and Y position of both eyes
xDiffBox = xLeftEyeBox - xRightEyeBox;
yDiffBox = yLeftEyeBox - yRightEyeBox;
xDiffCenter = referenceDistance(1) - (xLeftEyeCenter + xRightEyeCenter);
yDiffCenter = referenceDistance(2) - (yLeftEyeCenter + yRightEyeCenter);

x = 1:totalFrameNumber;
figure; plot(x, xLeftEyeBox, x, yLeftEyeBox); title('Left Eye'); xlabel('Frame'); ylabel('Distance in Pixels'); legend('X BOX', 'Y BOX');
figure; plot(x, xRightEyeBox, x, yRightEyeBox); title('Right Eye'); xlabel('Frame'); ylabel('Distance in Pixels'); legend('X BOX', 'Y BOX');

figure; plot(x, xLeftEyeCenter, x, yLeftEyeCenter); title('Left Eye'); xlabel('Frame'); ylabel('Distance in Pixels'); legend('X CENTER', 'Y CENTER');
figure; plot(x, xRightEyeCenter, x, yRightEyeCenter); title('Right Eye'); xlabel('Frame'); ylabel('Distance in Pixels'); legend('X CENTER', 'Y CENTER');

figure; plot(x, xDiffBox, x, yDiffBox); title('Difference BOX'); xlabel('Frame'); ylabel('Distance in Pixels'); legend('Difference X', 'Difference Y');
figure; plot(x, xDiffCenter, x, yDiffCenter); title('Difference CENTER'); xlabel('Frame'); ylabel('Distance in Pixels'); legend('Difference X', 'Difference Y');