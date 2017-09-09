%clear all; close all; clc;

%% Load Video file
%path = 'F:\MDS\CG\';
%path = '';

%filename = 'uncharted4first';
%filename = 'uncharted4second';
%filename = 'badcg2';
%filename = 'witcher';
%filename = 'mds_project_mad';
%filename = 'mds_project_xxx';
%filename = 'mds_project_cose';
%filename = 'mds_project_hard';
%filename = 'mds_project_ooo';
%filename = 'mds_project';
%filename = 'mds_project_still';
%filename = 'cg_bad';
%filename = 'test_oscillazioni_2secondi';
%filename = 'cg_bad';
%filename = 'pollomega';
%filename = 'megapollo_cg';
%filename = 'Activision R&D Real-time Character Demo-l6R6N4Vy0nE';
%ext = '.mp4';
%ext = '.mov';

videoFileReader = vision.VideoFileReader(strcat(strcat(path, filename), ext));
videoForFrameCount = VideoReader(strcat(strcat(path, filename), ext));
lastFrame = read(videoForFrameCount, inf);
totalFrameNumber = videoForFrameCount.NumberOfFrames;

% skip 1: clear iris
% skip 40: half iris
% skip 60: iris on topleft corner
% 300 470 548
skipFrames = 1;
maxFrames = 60; % Frames for each clip

for i = 1:skipFrames
    videoFrame      = step(videoFileReader);
end

clusters = 6;
% image = rgb2gray(videoFrame);
% image = imadjust(image);
%[leftEye, rightEye, leftEyePupil, leftIris, rightEyePupil, rightIris] = DetectionHelper.recoverPointsFromScratch(videoFrame, clusters);
[leftEye, rightEye, leftEyePupil, leftIris, rightEyePupil, rightIris, face_of_interest] = DetectionHelper.recoverPointsFromScratch(videoFrame, clusters);

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
pointTracker = vision.PointTracker('MaxBidirectionalError', 2, 'NumPyramidLevels', 3, 'BlockSize', [31 31]);

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
bboxLeftEye = [];
bboxRightEye = [];

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

% Angle of face rotation
eyeCenterRotation = [0, 0];
dirvectorBase = [-1, 0];
dirvectorEyes = [];
if (size(rightEyePupil, 1) > 0) && (size(leftEyePupil, 1) > 0)
    eyeCenterRotation = mean([leftEyePupil; rightEyePupil]);
    dirvectorEyes = rightEyePupil - eyeCenterRotation;
end
faceAngle = 0;
if size(dirvectorEyes) == size(dirvectorBase)
    faceAngle = acos(dot(dirvectorBase, dirvectorEyes) / norm(dirvectorBase) /norm(dirvectorEyes));
end

retryLeftCount = 0;
retryRightCount = 0;
retryMax = 30; % TODO A second, check the framerate

%% Store output valeus
outputValues = zeros(uint8(totalFrameNumber / maxFrames), 19);
currentClip = 1;
goodFrames = 0;

%% Compute circle metric
foundFace = 0;
framesToAverage = maxFrames;
arrayometrics = zeros(framesToAverage, 2);
metricCount = 1;
videoForAveraging = vision.VideoFileReader(strcat(strcat(path, filename), ext));

%for i = 1:600
%    videoFrameAvg      = step(videoForAveraging);
%end

%% Play the video and compute metrics for the circles
% To avoid choosing low intensity circles in some videos
videoPlayerAvg  = vision.VideoPlayer('Position',...
    [100 100 [size(videoFrame, 2), size(videoFrame, 1)]+30], 'Name', 'Calculating Metrics...');

while ~isDone(videoForAveraging)
    % get the next frame
    videoFrameAvg = step(videoForAveraging);
    
    if ~foundFace
        [leftEye, rightEye, leftEyePupil, leftIris, rightEyePupil, rightIris, face_of_interest, metrics] = DetectionHelper.recoverPointsFromScratch(videoFrameAvg, 0);
        if size(face_of_interest, 1) > 0
           % face found
           foundFace = 1;
        end
    end
    
    if foundFace
        [leftEye, rightEye, leftEyePupil, leftIris, rightEyePupil, rightIris, face_of_interest, metrics] = DetectionHelper.recoverPointsFromScratch(videoFrameAvg, 0);
        arrayometrics(metricCount, 1:2) = metrics;
        metricCount = metricCount + 1;
    end
    
    if metricCount > framesToAverage
        break;
    end
    
    step(videoPlayerAvg, videoFrameAvg);
end
averageMetrics = mean(arrayometrics);
averageMetric = mean(averageMetrics);
closepreview

%% Play the video and track both eyes
videoPlayer  = vision.VideoPlayer('Position',...
    [100 100 [size(videoFrame, 2), size(videoFrame, 1)]+30], 'Name', 'Tracking eyes...');

% Start the video
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
        [leftEye, rightEye, leftEyePupil, leftIris, rightEyePupil, rightIris, face_of_interest] = DetectionHelper.recoverPointsFromScratch(videoFrame, averageMetric);
        disp('Recovering all the points');
        goodFrames = 0;
        
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
    end
    
    if isFound(1) == 0 || ~leftEyeTracked % LeftEyePupil lost tracking
        leftEyeTracked = false;
        disp('Recovering Left Eye and Pupil using existing boxes');
        if retryLeftCount < retryMax
            %% Recover points knowing last bounding box
            % (try this for a few frames, then try and recover the whole face)
            [leftEyePupil, ~, ~, ~] = DetectionHelper.recoverPoints(videoFrame, bboxLeftEye, [], averageMetric);
            if size(leftEyePupil, 1) > 0
                points(1, :) = leftEyePupil;
                leftEyeTracked = true;
            end
            retryLeftCount = retryLeftCount + 1;
        else
            %% Recover points knowing nothing
            [leftEye, ~, leftEyePupil, ~, ~, ~, face_of_interest] = DetectionHelper.recoverPointsFromScratch(videoFrame, averageMetric, 1);
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
            [~, ~, rightEyePupil, ~] = DetectionHelper.recoverPoints(videoFrame, [], bboxRightEye, averageMetric);
            if size(rightEyePupil, 1) > 0
                points(2, :) = rightEyePupil;
                rightEyeTracked = true;
            end
            retryRightCount = retryRightCount + 1;
        else
            %%Recover points knowing nothing
            [~, rightEye, ~, ~, rightEyePupil, ~, face_of_interest] = DetectionHelper.recoverPointsFromScratch(videoFrame, averageMetric, 2);
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

    % same checks for the bounding boxes
    if size(isFound, 1) >= 6 && (prod(isFound(3:6)) == 0 || ~leftBoxTracked) && ~leftEyeTracked
        leftBoxTracked = false;
        [leftEye, ~, leftEyePupil, ~, ~, ~, face_of_interest] = DetectionHelper.recoverPointsFromScratch(videoFrame, averageMetric, 1);
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

    if size(isFound, 1) >= 10 && (prod(isFound(7:10)) == 0 || ~rightBoxTracked) && ~rightEyeTracked
        rightBoxTracked = false;
        [~, rightEye, ~, ~, rightEyePupil, ~, face_of_interest] = DetectionHelper.recoverPointsFromScratch(videoFrame, averageMetric, 2);
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

    % same checks for the "center" point
    if size(isFound, 1) >= 11 && isFound(11) == 0 && leftEyeTracked && rightEyeTracked
        disp('Recovering Center Point');
        eyeCenter = mean([bboxPointsLeft;bboxPointsRight]);
        referenceDistance = abs(mean(bboxPointsLeft) - mean(bboxPointsRight));
    end

    if size(isFound, 1) >= 11 && leftEyeTracked && rightEyeTracked
        eyeCenterRotation = mean([points(1, 1:2); points(2, 1:2)]);

        dirvectorEyes = [points(2, 1), points(2, 2)] - eyeCenterRotation;
        if size(dirvectorEyes) == size(dirvectorBase)
            faceAngle = acos(dot(dirvectorBase, dirvectorEyes) / norm(dirvectorBase) /norm(dirvectorEyes));
        end

        %% Calculating X and Y movement of pupils
        % Using Eye bounding boxes
        leftBoxCenter = mean(points(3:6, :));
        xLeftEyeBox(frameCount) = (leftBoxCenter(1) - points(1, 1)) * cos(faceAngle);
        yLeftEyeBox(frameCount) = (leftBoxCenter(2) - points(1, 2)) * sin(faceAngle + pi/2);

        rightBoxCenter = mean(points(7:10, :));
        xRightEyeBox(frameCount) = (rightBoxCenter(1) - points(2, 1)) * cos(faceAngle);
        yRightEyeBox(frameCount) = (rightBoxCenter(2) - points(2, 2)) * sin(faceAngle + pi/2);

        %Using Point between eyes
        pupilCenter = points(11, :);
        xLeftEyeCenter(frameCount) = (pupilCenter(1) - points(1, 1)) * cos(faceAngle);
        yLeftEyeCenter(frameCount) = (pupilCenter(2) - points(1, 2)) * sin(faceAngle + pi/2);

        xRightEyeCenter(frameCount) = (pupilCenter(1) - points(2, 1)) * cos(faceAngle);
        yRightEyeCenter(frameCount) = (pupilCenter(2) - points(2, 2)) * sin(faceAngle + pi/2);

        %disp('OK frame');
        %xLeftEyeCenter(frameCount)
    else
        disp('Throwaway frame');
        xLeftEyeBox(frameCount) = inf;
        yLeftEyeBox(frameCount) = inf;
        xRightEyeBox(frameCount) = inf;
        yRightEyeBox(frameCount) = inf;
        xLeftEyeCenter(frameCount) = inf;
        yLeftEyeCenter(frameCount) = inf;
        xRightEyeCenter(frameCount) = inf;
        yRightEyeCenter(frameCount) = inf;
        %xLeftEyeCenter(frameCount)
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
        videoFrame = insertObjectAnnotation(videoFrame, 'Rectangle', face_of_interest, 'Face');
    catch
        disp('Failed bboxes');
    end

    % Display tracked points
    videoFrame = insertMarker(videoFrame, points, '+', ...
        'Color', 'white');

    videoFrame = insertObjectAnnotation(videoFrame, 'rectangle', [0,0,50,50], faceAngle);
    videoFrame  = insertShape(videoFrame, 'Line', [points(1, 1), points(1, 2), points(2, 1), points(2, 2)],...
    'Color', {'green'},'Opacity',0.7);

    % Reset the points
    oldPoints = points;
    oldPoints(oldPoints < 0) = 1;
    setPoints(pointTracker, oldPoints);
    %end

    %% Compute statistics for learning/classification
    goodFrames = goodFrames + 1;
    if goodFrames > maxFrames
        % Store output and increment clip
        
        % Values: mean_sx, mean_dx, std_sx, std_dx
        bound_low = frameCount - maxFrames;
        bound_up = frameCount;
        
        box_xmean_sx = mean(xLeftEyeBox(bound_low:bound_up));
        box_ymean_sx = mean(yLeftEyeBox(bound_low:bound_up));
        
        box_xmean_dx = mean(xRightEyeBox(bound_low:bound_up));
        box_ymean_dx = mean(yRightEyeBox(bound_low:bound_up));
        
        center_xmean_sx = mean(xLeftEyeCenter(bound_low:bound_up));
        center_ymean_sx = mean(yLeftEyeCenter(bound_low:bound_up));
        
        center_xmean_dx = mean(xRightEyeCenter(bound_low:bound_up));
        center_ymean_dx = mean(yRightEyeCenter(bound_low:bound_up));
        
        
        box_xstd_sx = std(xLeftEyeBox(bound_low:bound_up));
        box_ystd_sx = std(yLeftEyeBox(bound_low:bound_up));
        
        box_xstd_dx = std(xRightEyeBox(bound_low:bound_up));
        box_ystd_dx = std(yRightEyeBox(bound_low:bound_up));
        
        center_xstd_sx = std(xLeftEyeCenter(bound_low:bound_up));
        center_ystd_sx = std(yLeftEyeCenter(bound_low:bound_up));
        
        center_xstd_dx = std(xRightEyeCenter(bound_low:bound_up));
        center_ystd_dx = std(yRightEyeCenter(bound_low:bound_up));
        
        outputValues(currentClip, 1:19) = [1, bound_low, bound_up,...
            box_xmean_sx, box_ymean_sx, box_xmean_dx, box_ymean_dx,...
            center_xmean_sx, center_ymean_sx, center_xmean_dx, center_ymean_dx,...
            box_xstd_sx, box_ystd_sx, box_xstd_dx, box_ystd_dx,...
            center_xstd_sx, center_ystd_sx, center_xstd_dx, center_ystd_dx];
        
        goodFrames = 0;
        currentClip = currentClip + 1;
    end

    %% Display the annotated video frame using the video player object
    step(videoPlayer, videoFrame);
end
closepreview

%% Remove useless frames
frameCount = frameCount - 1;

xLeftEyeBox = xLeftEyeBox(1:frameCount);
xRightEyeBox = xRightEyeBox(1:frameCount);
yLeftEyeBox = yLeftEyeBox(1:frameCount);
yRightEyeBox = yRightEyeBox(1:frameCount);

xLeftEyeCenter = xLeftEyeCenter(1:frameCount);
xRightEyeCenter = xRightEyeCenter(1:frameCount);
yLeftEyeCenter = yLeftEyeCenter(1:frameCount);
yRightEyeCenter = yRightEyeCenter(1:frameCount);

xLeftEyeBox(xLeftEyeBox == inf) = [];
xRightEyeBox(xRightEyeBox == inf) = [];
yLeftEyeBox(yLeftEyeBox == inf) = [];
yRightEyeBox(yRightEyeBox == inf) = [];

xLeftEyeCenter(xLeftEyeCenter == inf) = [];
xRightEyeCenter(xRightEyeCenter == inf) = [];
yLeftEyeCenter(yLeftEyeCenter == inf) = [];
yRightEyeCenter(yRightEyeCenter == inf) = [];

%% Plot X and Y position of both eyes
xDiffBox = xLeftEyeBox - xRightEyeBox;
yDiffBox = yLeftEyeBox - yRightEyeBox;
xDiffCenter = referenceDistance(1) - (xLeftEyeCenter + xRightEyeCenter);
yDiffCenter = yLeftEyeCenter - yRightEyeCenter;

pixelRange = 120;


%% Folder creation
if exist('video_class', 'var') == 0
    return;
end

output_path = sprintf('3.Results/%s/%s_output', video_class, filename);
mkdir(output_path);

%% Save and show plots
x = 1:size(xLeftEyeCenter, 1);
figure;
subplot(3, 2, 1);
plot(x, xLeftEyeBox, x, yLeftEyeBox); title('Left Eye'); xlabel('Frame'); ylabel('Distance in Pixels'); legend('X BOX', 'Y BOX');
axis([1 60 -pixelRange pixelRange]);
grid on
%savefig(sprintf('generated/%s_left_eye_box.fig', filename));
subplot(3, 2, 2);
plot(x, xRightEyeBox, x, yRightEyeBox); title('Right Eye'); xlabel('Frame'); ylabel('Distance in Pixels'); legend('X BOX', 'Y BOX');
axis([1 60 -pixelRange pixelRange]);
grid on
%savefig(sprintf('generated/%s_right_eye_box.fig', filename));

subplot(3, 2, 3);
plot(x, xLeftEyeCenter, x, yLeftEyeCenter); title('Left Eye'); xlabel('Frame'); ylabel('Distance in Pixels'); legend('X CENTER', 'Y CENTER');
axis([1 60 -pixelRange pixelRange]);
grid on
%savefig(sprintf('generated/%s_left_eye_center.fig', filename));
subplot(3, 2, 4);
plot(x, xRightEyeCenter, x, yRightEyeCenter); title('Right Eye'); xlabel('Frame'); ylabel('Distance in Pixels'); legend('X CENTER', 'Y CENTER');
axis([1 60 -pixelRange pixelRange]);
grid on
%savefig(sprintf('generated/%s_right_eye_center.fig', filename));

subplot(3, 2, 5);
plot(x, xDiffBox, x, yDiffBox); title('Difference BOX'); xlabel('Frame'); ylabel('Distance in Pixels'); legend('Difference X', 'Difference Y');
axis([1 60 -pixelRange pixelRange]);
grid on
%savefig(sprintf('generated/%s_diff_box.fig', filename));
subplot(3, 2, 6);
plot(x, xDiffCenter, x, yDiffCenter); title('Difference CENTER'); xlabel('Frame'); ylabel('Distance in Pixels'); legend('Difference X', 'Difference Y');
axis([1 60 -pixelRange pixelRange]);
grid on
savefig(strcat(output_path, '/charts.fig'));

figure;
subplot(2, 2, 1);
scatter(xLeftEyeBox, yLeftEyeBox); title('Left Eye using BOX'); xlabel('X in pixels'); ylabel('Y in pixels');
line(xLeftEyeBox, yLeftEyeBox,'Color','red');
axis([-120 120 -120 120]);
grid on
subplot(2, 2, 2);
scatter(xRightEyeBox, yRightEyeBox); title('Right Eye using BOX'); xlabel('X in pixels'); ylabel('Y in pixels');
axis([-120 120 -120 120]);
line(xRightEyeBox, yRightEyeBox,'Color','red');
grid on

subplot(2, 2, 3);
scatter(xLeftEyeCenter, yLeftEyeCenter); title('Left Eye using CENTER'); xlabel('X in pixels'); ylabel('Y in pixels');
axis([-120 120 -120 120]);
line(xLeftEyeCenter, yLeftEyeCenter,'Color','red');
grid on
subplot(2, 2, 4);
scatter(xRightEyeCenter, yRightEyeCenter); title('Right Eye using CENTER'); xlabel('X in pixels'); ylabel('Y in pixels');
axis([-120 120 -120 120]);
line(xRightEyeCenter, yRightEyeCenter,'Color','red');
grid on
savefig(strcat(output_path, '/scatter.fig'));

%% Save data
save(strcat(output_path, '/features.mat'), 'outputValues');