%% PUPILHELPER
%  Class that contains helper functions for pupil detection using
%  morphological filters
classdef PupilHelper
    methods (Static) 
        
        %% FINDPUPIL
        %  Function used to recover all the points.
        %  INPUT:
        %         - videoFrame: current video frame
        %         - eyeBox : bounding box of the eye
        %         - metricth: threshold for circle strength
        %         - debug: true or false, to show debugging images
        %                   slows down everything by a lot, recommended to
        %                   run for just a few frames if set to true
        %  OUTPUT:
        %         - eyePupil : Left pupil
        %         - irisResults : Left iris
        %         - success : true if everything ran correctly, false
        %                       otherwise
        %         - maxmetrics : maximum circle strength found
        function[eyePupil, irisResults, success, maxmetric] = findPupil(videoFrame, eyeBox, metricth, debug)
            maxmetric = 0;
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
            if debug
                figure; imshow(eyeImage, 'InitialMagnification', 'fit'); title('cropped image');
            end
            
            [n, m] = size(eyeImage);
            if m == 0 || n == 0
                success = false;
                eyePupil = [];
                irisResults = [];
                return;
            end
            
            % Lavoriamo sull'immagine dell'occhio in bianco e nero per
            % semplicita' (la pupilla dovrebbe essere la parte piu' scura)
            eyeImage = rgb2gray(eyeImage);
          
            % imadjust default per aggiustare la luminosita'
            eyeImage = imadjust(eyeImage);
            if debug
                %figure; imshow(eyeImage, 'InitialMagnification', 'fit'); title(strcat('imadjust default',eyeString));
            end
            
            % imadjust per mantenere solo le parti piu' scure
            eyeImage = imadjust(eyeImage, [0.0 0.25], [0 1]);
            %eyeImage = imadjust(eyeImage, [0.25 0.3], [0.5 1]);
            %if debug
            %    figure; imshow(eyeImage, 'InitialMagnification', 'fit'); title(strcat('imadjust1',eyeString));
            %end
            %eyeImage = imadjust(eyeImage, [0 0.05], [0 1]);
            %if debug
            %    figure; imshow(eyeImage, 'InitialMagnification', 'fit'); title(strcat('imadjust2',eyeString));
            %end
            
            eyeImage = histeq(eyeImage);
            if debug
                figure; imshow(eyeImage, 'InitialMagnification', 'fit'); title('histeq');
            end
            eyeImage = imbinarize(eyeImage, 0.3);
            if debug
                figure; imshow(eyeImage, 'InitialMagnification', 'fit'); title('imadjust2');
            end
            eyeImage = uint8(255 * eyeImage);
            
            se = [strel('disk', 2)];
            eyeImage = imerode(eyeImage, se);
            se = [strel('diamond', 3)];
            eyeImage = imdilate(eyeImage, se);
            
            if debug
                figure; imshow(eyeImage, 'InitialMagnification', 'fit'); title('imerode');
            end
            
            % imbothat: rimuoviamo le linee lunge circa 20 pixel
            se = strel('line', n* 4/5, 0);
            bothat1 = imbothat(eyeImage, se);
            if debug
                figure; imshow(imcomplement(bothat1), 'InitialMagnification', 'fit'); title('bothat1');
            end
            
            se = strel('line', n * 4/5, 15);
            bothat2 = imbothat(eyeImage, se);
            if debug
                figure; imshow(imcomplement(bothat2), 'InitialMagnification', 'fit'); title('bothat2');
            end
            
            se = strel('line', n * 4/5, -15);
            bothat3 = imbothat(eyeImage, se);
            if debug
                figure; imshow(imcomplement(bothat3), 'InitialMagnification', 'fit'); title('bothat3');
            end
            
            % il risultato di imbothat e' quello che ci aspettiamo ma con
            % bianchi e neri invertiti: complementiamo l'immagine
            eyeImage = imcomplement(eyeImage);
            
            eyeImage = imsubtract(eyeImage, imcomplement(bothat1));
            eyeImage = imsubtract(eyeImage, imcomplement(bothat2));
            eyeImage = imsubtract(eyeImage, imcomplement(bothat3));
            if debug
                figure; imshow(eyeImage, 'InitialMagnification', 'fit'); title('imsubtract');
            end
            
            % imerode per erodere utilizzando una linea verticale di due
            % pixel come structural element (due volte per essere sicuri di
            % rimuovere tutto tranne la pupilla)
            se = [strel('line', 2, 90)];
            eyeImage = imerode(eyeImage, se);
            if debug
                figure; imshow(eyeImage, 'InitialMagnification', 'fit'); title('imerode');
            end
            
            se = strel('disk', 1);
            toremove = imtophat(eyeImage, se);
            if debug
                figure; imshow(imcomplement(toremove), 'InitialMagnification', 'fit'); title('tophat');
            end
            
            % imdilate per ripristinare alcuni pixel della pupilla
            %se = [strel('disk', 1)];
            %eyeImage = imdilate(eyeImage, se);
            
            
            %eyeImage = imcomplement(eyeImage);
            %if debug
            %    figure; imshow(eyeImage, 'InitialMagnification', 'fit'); title(strcat('imdilate',eyeString));
            %end
            
            % Creazione di una immagine che contiene solo dischi di raggio
            % minore di 2 pixel
            %se = strel('disk', 2);
            %toremove = imtophat(eyeImage, se);
            %if debug
            %    figure; imshow(eyeImage, 'InitialMagnification', 'fit'); title(strcat('original',eyeString));
            %    figure; imshow(toremove, 'InitialMagnification', 'fit'); title(strcat('imtophat',eyeString));
            %end
            
            % Rimuoviamo tutti i dischi di raggio minore di 2 pixel
            % all'interno dell'immagine dell'occhio
            eyeImage = imcomplement(imsubtract(eyeImage, toremove));
            
            eyeImage = im2bw(eyeImage, 0.9);
            
            if debug
                figure; imshow(eyeImage, 'InitialMagnification', 'fit'); title('subtract');
            end
            
            minRadius = floor(n/16);
            maxRadius = floor(n/4);
            
            minRadius
            maxRadius
            if minRadius <= 0 || maxRadius <= 0
                success = false;
                eyePupil = [];
                irisResults = [];
                return;
            end
            
            %minRadius
            %maxRadius
            %eyeImage
            [centers, radii, metric] = imfindcircles(eyeImage,[minRadius maxRadius],'ObjectPolarity','dark', 'Sensitivity',0.9);
            %[centers, radii] = imfindcircles(eyeImage,[minRadius maxRadius], 'Sensitivity',0.9);
            %centers
            %radii
            
            if ((size(centers, 1) == 0) || (size(radii, 1) == 0))
                success = false;
                eyePupil = [];
                irisResults = [];
                return;
            end
            [maxradius, argmradius] = max(radii);
            [maxmetric, argmmetric] = max(metric);
            metric
            
            if maxmetric < metricth
                success = false;
                eyePupil = [];
                irisResults = [];
                return;
            end
            
            %maxcircle = centers(argmradius,:);
            maxcircle = centers(argmmetric,:);
            maxradius = radii(argmmetric,:);
            
            if debug
                h = viscircles(centers,radii);
                %h
            end

            
             eyeBox = double(eyeBox);
             %eyeBox
             %maxcircle
             %eyeBox(1, 1:2)
             eyePupil = eyeBox(1, 1:2) + maxcircle;
             %eyePupil
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

        %% unused
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
                [leftEyePupil, leftIris, successL] = PupilHelper.findPupil(videoFrame, leftEye, 'left', debug);
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
                [rightEyePupil, rightIris, successR] = PupilHelper.findPupil(videoFrame, rightEye, 'right', debug);
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
        
    end
end