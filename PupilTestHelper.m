classdef PupilTestHelper
    methods (Static) 
       
        % eye by default(0) means both, 1 means left eye, 2 means right eye
        function[leftEye, rightEye, leftEyePupil, leftIris, rightEyePupil, rightIris] = recoverPointsFromScratch(videoFrame, eye)
            successL = true; successR = true; leftEyePupil = [];
            rightEyePupil = []; leftIris = []; rightIris = [];
            leftEye = []; rightEye = [];
            if nargin < 3
                eye = 0;
            end
            
            faceDetector = vision.CascadeObjectDetector();
            eyeDetector = vision.CascadeObjectDetector('EyePairSmall', 'UseROI', true);
            leftEyeDetector = vision.CascadeObjectDetector('LeftEye', 'UseROI', true);
            rightEyeDetector = vision.CascadeObjectDetector('RightEye', 'UseROI', true);
            eyeBigDetector = vision.CascadeObjectDetector('EyePairBig', 'UseROI', true);
            %noseDetector = vision.CascadeObjectDetector('Nose', 'UseROI', true);
            
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
            
            debug = false;
            
            %% Eye finding
            % Need some processing to find the correct Left Eye and Right Eye
            % by using the "eyes" Bounding Box, and then picking the best box
            if eye == 0 || eye == 1
                %leftEyes = SupportFunctions.removeNonIntersecting(leftEyes, eyes, threshold);
                %if size(leftEyes, 1) > 0
                %    leftEye = SupportFunctions.getRightMost(leftEyes);
                %else
                %    leftEye = SupportFunctions.getRightMost(rightEyes);
                %end
                %m = size(leftIndexes, 2);
                %leftEye = SupportFunctions.getRightMost(leftEyes(leftIndexes(1:min(2, m)), :));
                leftEye = SupportFunctions.getRightMost(totalEyes);
                [leftEyePupil, leftIris, successL] = PupilTestHelper.findPupil(videoFrame, leftEye, 'left', debug);
            end
            if eye == 0 || eye == 2
                %rightEyes = SupportFunctions.removeNonIntersecting(rightEyes, eyes, threshold);
                %if size(rightEyes, 1) > 0
                %    rightEye = SupportFunctions.getLeftMost(rightEyes);
                %else
                %    rightEye = SupportFunctions.getLeftMost(leftEyes);
                %end
                %m = size(rightIndexes, 2);
                %rightEye = SupportFunctions.getLeftMost(rightEyes(rightIndexes(1:min(2, m)), :));
                rightEye = SupportFunctions.getLeftMost(totalEyes);
                [rightEyePupil, rightIris, successR] = PupilTestHelper.findPupil(videoFrame, rightEye, 'right', debug);
            end
            
            if debug
                videoFrameShow = insertObjectAnnotation(videoFrame, 'Rectangle', face_of_interest, 'Face');
                videoFrameShow = insertObjectAnnotation(videoFrameShow, 'Rectangle', eyes, 'Eyes');
                videoFrameShow = insertObjectAnnotation(videoFrameShow, 'Rectangle', leftEyes, 'Left Eye');
                videoFrameShow = insertObjectAnnotation(videoFrameShow, 'Rectangle', rightEyes, 'Right Eye');
                imshow(videoFrameShow);
            end
            
            if eye == 0 && successL && successR %Only if getting both eyes
                [successL, successR] = DetectionHelper.checkOverlap(leftEye, rightEye);
            end
            
            if ~successL
                leftEyePupil = [];
            end
            if ~successR
                rightEyePupil = [];
            end
        end
        
        function[eyePupil, irisResults, success] = findPupil(videoFrame, eyeBox, eyeString, debug)
            if nargin < 4
                debug = false;
            end
            success = true;
            
            if size(eyeBox, 1) == 0
                success = false;
                eyePupil = [];
                irisResults = [];
                return;
            end
  
            eyeImage = imcrop(videoFrame, eyeBox);
%             figure; imshow(eyeImage, 'InitialMagnification', 'fit'); title('cropped image');

            [n, m] = size(eyeImage);
            if m == 0 || n == 0
                success = false;
                eyePupil = [];
                irisResults = [];
                return;
            end
            
            eyeImage = rgb2gray(eyeImage);
%             figure; imshow(eyeImage, 'InitialMagnification', 'fit'); title('gray scale image');
            
          
            eyeImage = imadjust(eyeImage);
%             figure; imshow(eyeImage, 'InitialMagnification', 'fit'); title('imadjust');
            
            eyeImage = imadjust(eyeImage, [0.2 0.25], [0 1]);
%             figure; imshow(eyeImage, 'InitialMagnification', 'fit'); title('imadjust cut');
            eyeImage = imadjust(eyeImage, [0 0.05], [0 1]);
%             figure; imshow(eyeImage, 'InitialMagnification', 'fit'); title(strcat('imadjust cut 2 ',eyeString));

            minRadius = floor(n/8);
            maxRadius = floor(n/2);
            minRadius
            maxRadius
            [centers, radii] = imfindcircles(eyeImage,[minRadius maxRadius],'ObjectPolarity','dark', 'Sensitivity',0.9)
            centers
            radii
            
            if ((size(centers, 1) == 0) || (size(radii, 1) == 0))
                success = false;
                eyePupil = [];
                irisResults = [];
                return;
            end
            maxcircle=centers(1,:);
            maxradius = radii(1);
            h = viscircles(centers,radii);
            h

            
             eyeBox = double(eyeBox);
             eyeBox
             maxcircle
             eyeBox(1, 1:2)
             eyePupil = eyeBox(1, 1:2) + maxcircle;
             eyePupil
             %irisLeft = [eyePupil(1) - maxradius eyePupil(2)];
             %irisRight = [eyePupil(1) + maxradius eyePupil(2)];
             
             numberOfPoints = 20;
             angles = linspace(0, 2*pi, numberOfPoints);
             f = @(theta) [cos(theta), sin(theta)];
             r = arrayfun(@(i)f(angles(i)), 1:numberOfPoints, 'UniformOutput', false);
             irisPoints = reshape(cell2mat(r), [2 numberOfPoints]);
             irisResults = zeros(numberOfPoints, 2);
             
             irisPoints = (irisPoints .* maxradius);
             for i = 1:numberOfPoints
                 irisResults(i, :) = irisPoints(:, i)' + maxcircle + eyeBox(1:2);
             end
        end

    end
end