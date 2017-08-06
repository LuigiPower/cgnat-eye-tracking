classdef DetectionHelper
    methods (Static)
        %% Point recovery
        function[leftEyePupil, leftIris, rightEyePupil, rightIris] = recoverPoints(videoFrame, bboxLeftEye, bboxRightEye, clusters)
            %% Eye finding
            successL = true; successR = true; leftEyePupil = [];
            rightEyePupil = []; leftIris = []; rightIris = [];
            if size(bboxLeftEye, 1) ~= 0
                [leftEyePupil, leftIris, successL] = PupilTestHelper.findPupil(videoFrame, int16(bboxLeftEye), clusters, false);
            end
            if size(bboxRightEye, 1) ~= 0
                [rightEyePupil, rightIris, successR] = PupilTestHelper.findPupil(videoFrame, int16(bboxRightEye), clusters, false);
            end
            if ~successL || ~successR
                leftEyePupil = [];
                rightEyePupil = [];
            end
        end
        
        % eye by default(0) means both, 1 means left eye, 2 means right eye
        function[leftEye, rightEye, leftEyePupil, leftIris, rightEyePupil, rightIris, face_of_interest] = recoverPointsFromScratch(videoFrame, clusters, eye)
            successL = true; successR = true; leftEyePupil = [];
            face_of_interest = [];
            rightEyePupil = []; leftIris = []; rightIris = [];
            leftEye = []; rightEye = [];
            if nargin < 3
                eye = 0;
            end
            
            faceDetector = vision.CascadeObjectDetector();
            eyeDetector = vision.CascadeObjectDetector('EyePairBig', 'UseROI', true);
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
            
            for i = 1:size(eyes, 1)
                eyes(indexes(i), :) = DetectionHelper.fitFace(eyes(indexes(1), :), face_of_interest);
            end
            
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
            nw = eyes(indexes(1), 3);
            nh = eyes(indexes(1), 4);
            newLeft = eyes(indexes(1), 1);
            newTop = eyes(indexes(1), 2);
            
            if nw > nh
                nw = nw/2
                newLeft = eyes(indexes(1), 1) + nw;
            else
                nh = nh/2
                newTop = eyes(indexes(1), 2) + nh;
            end
            
            if eye == 0 || eye == 1
                %leftEyes = SupportFunctions.removeNonIntersecting(leftEyes, eyes, threshold);
                %if size(leftEyes, 1) > 0
                %    leftEye = SupportFunctions.getRightMost(leftEyes);
                %else
                %    leftEye = SupportFunctions.getRightMost(rightEyes);
                %end
                %m = size(leftIndexes, 2);
                %leftEye = SupportFunctions.getRightMost(leftEyes(leftIndexes(1:min(2, m)), :));
                %leftEye = SupportFunctions.getRightMost(totalEyes);
                
                tempLeftEye = [newLeft, newTop, nw, nh];
                leftEyes = leftEyeDetector(videoFrame, tempLeftEye);
                [~, leftIndexes] = SupportFunctions.orderDescByArea(leftEyes);
                leftEye = leftEyes(leftIndexes(1), :);
                [leftEyePupil, leftIris, successL] = PupilTestHelper.findPupil(videoFrame, leftEye, clusters, debug);
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
                %rightEye = SupportFunctions.getLeftMost(totalEyes);
                
                tempRightEye = [eyes(indexes(1), 1), eyes(indexes(1), 2), nw, nh];
                rightEyes = rightEyeDetector(videoFrame, tempRightEye);
                [~, rightIndexes] = SupportFunctions.orderDescByArea(rightEyes);
                rightEye = rightEyes(rightIndexes(1), :);
                [rightEyePupil, rightIris, successR] = PupilTestHelper.findPupil(videoFrame, rightEye, clusters, debug);
            end
            
            if debug
                videoFrameShow = insertObjectAnnotation(videoFrame, 'Rectangle', face_of_interest, 'Face');
                videoFrameShow = insertObjectAnnotation(videoFrameShow, 'Rectangle', eyes, 'Eyes');
                videoFrameShow = insertObjectAnnotation(videoFrameShow, 'Rectangle', leftEyes, 'Left Eye');
                videoFrameShow = insertObjectAnnotation(videoFrameShow, 'Rectangle', rightEyes, 'Right Eye');
                imshow(videoFrameShow);
            end
            
            if eye == 0 && size(leftEye, 1) > 0 && size(rightEye, 1) > 0
                [successL, successR, leftEye, rightEye] = DetectionHelper.checkOverlap(leftEye, rightEye);
                if successL && successR
                    %[leftEye, rightEye] = DetectionHelper.enlargeEyes(leftEye, rightEye);
                    [leftEyePupil, leftIris, successL] = PupilTestHelper.findPupil(videoFrame, leftEye, clusters, debug);
                    [rightEyePupil, rightIris, successR] = PupilTestHelper.findPupil(videoFrame, rightEye, clusters, debug);
                end
            end
            
            if ~successL
                leftEyePupil = [];
            end
            if ~successR
                rightEyePupil = [];
            end
        end
        
        function[newEyes] = fitFace(eyes, face)
            disp('Fitting face');
            center = [eyes(1) + eyes(3)/2, eyes(2) + eyes(4)/2];
            w = eyes(3);
            h = eyes(4);

            for scale = 1.0:0.05:2.0
                nw = w * scale;
                nh = h * scale;
                newEyes = [center(1) - nw/2, center(2) - nh/2, nw, nh];
                
                if nw > face(3)
                    break
                end
            end
        end
       
        function[newLeftEye, newRightEye] = enlargeEyes(leftEye, rightEye)
            newLeftEye = leftEye; newRightEye = rightEye;
            disp('Enlarging eyes');
            
            leftCenter = [leftEye(1) + leftEye(3)/2, leftEye(2) + leftEye(4)/2];
            rightCenter = [rightEye(1) + rightEye(3)/2, rightEye(2) + rightEye(4)/2];
            leftW = leftEye(3);
            leftH = leftEye(4);
            rightW = rightEye(3);
            rightH = rightEye(4);

            leftEye
            rightEye
            leftCenter
            rightCenter

            for scale = 1.0:0.05:5.0
                disp('in the for');
                lnw = leftW * scale;
                lnh = leftH * scale;
                rnw = rightW * scale;
                rnh = rightW * scale;
                newLeftEye = [leftCenter(1) - lnw/2, leftCenter(2) - lnh/2, lnw, lnh];
                newRightEye = [rightCenter(1) - rnw/2, rightCenter(2) - rnh/2, rnw, rnh];

                overlap = bboxOverlapRatio(newLeftEye, newRightEye);
                if overlap > 0
                    newLeftEye
                    newRightEye
                    disp('breaking');
                    break
                end
            end
        end
        
        function[successL, successR, newLeftEye, newRightEye] = checkOverlap(leftEye, rightEye)
            successL = true; successR = true;
            newLeftEye = leftEye; newRightEye = rightEye;
            
            overlap = bboxOverlapRatio(leftEye, rightEye);
            if overlap > 0
                successL = false;
                successR = false;
            end
        end
        
        %% toGrayScale
        function[image] = toGrayScale(image)
            % toGrayScale
            % Creates 3 channel gray scale
            %   @param image image to convert
            
            imageGS = imadjust(rgb2gray(image), [0.0 0.18], [0 1.0]);
            image(:, :, 1) = imageGS;
            image(:, :, 2) = imageGS;
            image(:, :, 3) = imageGS;
            %figure, imshow(imageGS), title('Adjusted GS');
        end
        
        %% Clustering
        function[cluster_idx, cluster_center, pixel_labels] = clusterImage(image, nColors)
            % clusterImage
            % Clusters image based on colors using kmeans,
            % works best on greyscale
            %   @param image image to cluster
            %   @param nColors number of clusters
            
            cform = makecform('srgb2lab');
            lab_image = applycform(double(image), cform);

            ab = double(lab_image(:, :, 2:3));
            nrows = size(ab, 1);
            ncols = size(ab, 2);
            ab = reshape(ab, nrows * ncols, 2);

            % repeat the clustering 3 times to avoid local minima
            [cluster_idx, cluster_center] = kmeans(ab, nColors, 'distance', ...
                                                       'sqEuclidean', 'Replicates', 5);

            pixel_labels = reshape(cluster_idx, nrows, ncols);
        end
        
        function[cluster_images, centers] = createClusterImages(nColors, pixel_labels)
            % createClusterImages
            % Creates Images based on labels for each cluster
            %   @param nColors number of clusters
            %   @param pixel_labels result of clustering
            
            [n, m] = size(pixel_labels);
            centers = zeros(nColors, 2);
            counts = zeros(nColors);
            cluster_images = zeros(n, m, nColors);
            cluster_images(1:n, 1:m, 1:nColors) = 0;

            for cluster = 1:nColors
                for i = 1:n
                    for j = 1:m
                        if(pixel_labels(i, j) == cluster)
                            centers(cluster, :) = centers(cluster, :) + [i, j];
                            counts(cluster) = counts(cluster) + 1;
                            cluster_images(i, j, cluster) = 255;
                        end
                    end
                end
                centers(cluster, :) = centers(cluster, :) ./ counts(cluster);
            end

            scrsz = get(groot,'ScreenSize');
            for index = 1:nColors
                %figure('Position', ...
                    %[(scrsz(3)/nColors)*index scrsz(4)/2 scrsz(3)/nColors scrsz(4)/2]), ...
                    %imshow(cluster_images(:, :, index)), ...
                    %title('image labeled by cluster index');
            end
        end
        
        %% Eye finding
        function[maxcircle, maxradius, maxcluster] = findMaxRegionpropCluster(cluster_images, lbradius, ubradius)
            % findMaxCircleCluster
            % Finds the biggest circle in the given images, and returns
            % it with the index of the related image
            %   @param cluster_images images to check
            %   @param lbradius Radius Lower Bound
            %   @param ubradius Radius Upper Bound
            
            image_count = size(cluster_images, 3);
            maxcircle = [0 0];
            maxradius = 0;
            maxcluster = 1;

            for i = 1:image_count
                [w, h] = size(cluster_images(:, :, i));
                %padding = h/4;
                padding = 0;
                searching = imcrop(cluster_images(:, :, i), [padding padding (w-padding) (h-padding)]);
                stats = regionprops('table', searching, 'Centroid',...
                                    'MajorAxisLength', 'MinorAxisLength');
                circleCenters = stats.Centroid;
                diameters = mean([stats.MajorAxisLength stats.MinorAxisLength],2);
                radii = diameters/2;
                [nc, ~] = size(circleCenters);
                if(nc > 0)
                    [mradius, argmradius] = max(radii);
                    if(mradius > maxradius && mradius < ubradius && mradius > lbradius)
                        maxcluster = i;
                        maxradius = max(radii);
                        maxcircle = circleCenters(argmradius, :) + padding;
                    end
                end
            end
        end
        
        function[maxcircle, maxradius, maxcluster] = findMaxCircleCluster(cluster_images, lbradius, ubradius)
            % findMaxCircleCluster
            % Finds the biggest circle in the given images, and returns
            % it with the index of the related image
            %   @param cluster_images images to check
            %   @param lbradius Radius Lower Bound
            %   @param ubradius Radius Upper Bound
            
            image_count = size(cluster_images, 3);
            maxcircle = [0 0];
            maxradius = 0;
            maxcluster = 1;

            for i = 1:image_count
                [w, h] = size(cluster_images(:, :, i));
                %padding = h/4;
                padding = ubradius;
                searching = imcrop(cluster_images(:, :, i), [padding padding (w-padding) (h-padding)]);
                [circleCenters, radii, ~] = imfindcircles(searching, uint16([lbradius ubradius]), 'ObjectPolarity', 'dark', 'Sensitivity', 0.93);
                [nc, ~] = size(circleCenters);
                if(nc > 0)
                    [mradius, argmradius] = max(radii);
                    if(mradius > maxradius)
                        maxcluster = i;
                        maxradius = max(radii);
                        maxcircle = circleCenters(argmradius, :) + padding;
                    end
                end
            end
        end
        
%         function[eyePupil, irisResults, success] = findEye(videoFrame, eyeBox, clusters, debug)
%             if nargin < 4
%                 debug = false;
%             end
%             success = true;
%             
%             if size(eyeBox, 1) == 0
%                 success = false;
%                 eyePupil = [];
%                 irisResults = [];
%                 return;
%             end
%   
%             eyeImage = imcrop(videoFrame, eyeBox);
%             [n, m] = size(eyeImage);
%             if m == 0 || n == 0
%                 success = false;
%                 eyePupil = [];
%                 irisResults = [];
%                 return;
%             end
%             
%             eyeImage = DetectionHelper.toGrayScale(eyeImage);
% 
%             [~, ~, pixel_labels] = DetectionHelper.clusterImage(...
%                 eyeImage, clusters);
% 
%             [cluster_images, ~] = DetectionHelper.createClusterImages(clusters, pixel_labels);
% 
%             [maxcircle, maxradius, maxcluster] = DetectionHelper.findMaxCircleCluster(cluster_images, m/60, m/20);
%             if maxradius == 0
%                 success = false;
%             end
% 
%             if(debug)
%                 figure; imshow(eyeImage);
%                 figure; imshow(cluster_images(:, :, maxcluster)); title('MAXCLUSTER');
%                 viscircles(maxcircle, maxradius, 'EdgeColor', 'r');
%             end
%             
%             eyeBox = double(eyeBox);
%             eyePupil = eyeBox(1, 1:2) + maxcircle;
%             %irisLeft = [eyePupil(1) - maxradius eyePupil(2)];
%             %irisRight = [eyePupil(1) + maxradius eyePupil(2)];
%             
%             numberOfPoints = 20;
%             angles = linspace(0, 2*pi, numberOfPoints);
%             f = @(theta) [cos(theta), sin(theta)];
%             r = arrayfun(@(i)f(angles(i)), 1:numberOfPoints, 'UniformOutput', false);
%             irisPoints = reshape(cell2mat(r), [2 numberOfPoints]);
%             irisResults = zeros(numberOfPoints, 2);
%             
%             irisPoints = (irisPoints .* maxradius);
%             for i = 1:numberOfPoints
%                 irisResults(i, :) = irisPoints(:, i)' + maxcircle + eyeBox(1:2);
%             end
%         end

function[eyePupil, irisResults, success] = findEye(videoFrame, eyeBox, clusters, debug)
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
%             figure; imshow(eyeImage, 'InitialMagnification', 'fit'); title('imadjust cut 2');

            [x,y,z] = size(eyeImage);
            x
            minRadius = floor(x/8);
            maxRadius = floor(x/2);
            minRadius
            maxRadius
            [centers, radii] = imfindcircles(eyeImage,[minRadius maxRadius],'ObjectPolarity','dark', 'Sensitivity',0.9)
            centers
            radii
            maxcircle=centers(1);
            maxradius = radii(1);
            h = viscircles(centers,radii);
            h

%              eyeImage = DetectionHelper.toGrayScale(eyeImage);

% 
%             [~, ~, pixel_labels] = DetectionHelper.clusterImage(...
%                 eyeImage, clusters);
% 
%             [cluster_images, ~] = DetectionHelper.createClusterImages(clusters, pixel_labels);
% 
%             [maxcircle, maxradius, maxcluster] = DetectionHelper.findMaxCircleCluster(cluster_images, m/60, m/20);
%             if maxradius == 0
%                 success = false;
%             end
% 
%             if(debug)
%                 figure; imshow(eyeImage);
%                 figure; imshow(cluster_images(:, :, maxcluster)); title('MAXCLUSTER');
%                 viscircles(maxcircle, maxradius, 'EdgeColor', 'r');
%             end
            
             eyeBox = double(eyeBox);
             eyePupil = eyeBox(1, 1:2) + maxcircle;
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
