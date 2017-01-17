classdef DetectionHelper
    methods (Static)
        %% Point recovery
        function[leftEyePupil, leftIris, rightEyePupil, rightIris] = recoverPoints(videoFrame, bboxLeftEye, bboxRightEye, clusters)
            %% Eye finding
            successL = true; successR = true; leftEyePupil = [];
            rightEyePupil = []; leftIris = []; rightIris = [];
            if size(bboxLeftEye, 1) ~= 0
                [leftEyePupil, leftIris, successL] = DetectionHelper.findEye(videoFrame, int16(bboxLeftEye), clusters);
            end
            if size(bboxRightEye, 1) ~= 0
                [rightEyePupil, rightIris, successR] = DetectionHelper.findEye(videoFrame, int16(bboxRightEye), clusters);
            end
            if ~successL || ~successR
                leftEyePupil = [];
                rightEyePupil = [];
            end
        end
        
        % eye by default(0) means both, 1 means left eye, 2 means right eye
        function[leftEye, rightEye, leftEyePupil, leftIris, rightEyePupil, rightIris] = recoverPointsFromScratch(videoFrame, clusters, eye)
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
            %eyeBigDetector = vision.CascadeObjectDetector('EyePairBig', 'UseROI', true);
            %noseDetector = vision.CascadeObjectDetector('Nose', 'UseROI', true);

            bbox = faceDetector(videoFrame);
            [~, indexes] = SupportFunctions.orderDescByArea(bbox);

            face_of_interest = bbox(indexes(1), :);
            eyes = eyeDetector(videoFrame, face_of_interest);

            threshold = 0.1;
            
            %% Eye finding
            % Need some processing to find the correct Left Eye and Right Eye
            % by using the "eyes" Bounding Box, and then picking the best box
            if eye == 0 || eye == 1
                leftEyes = leftEyeDetector(videoFrame, face_of_interest);
                leftEyes = SupportFunctions.removeNonIntersecting(leftEyes, eyes, threshold);
                leftEye = SupportFunctions.getRightMost(leftEyes);
                [leftEyePupil, leftIris, successL] = DetectionHelper.findEye(videoFrame, leftEye, clusters);
            end
            if eye == 0 || eye == 2
                rightEyes = rightEyeDetector(videoFrame, face_of_interest);
                rightEyes = SupportFunctions.removeNonIntersecting(rightEyes, eyes, threshold);
                rightEye = SupportFunctions.getLeftMost(rightEyes);
                [rightEyePupil, rightIris, successR] = DetectionHelper.findEye(videoFrame, rightEye, clusters);
            end
            
            if ~successL || ~successR
                leftEyePupil = [];
                rightEyePupil = [];
            end
        end
        
        %% toGrayScale
        function[image] = toGrayScale(image)
            % toGrayScale
            % Creates 3 channel gray scale
            %   @param image image to convert
            
            imageGS = imadjust(rgb2gray(image), [0.1 0.25], [0 1.0]);
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
                padding = 0;
                searching = imcrop(cluster_images(:, :, i), [padding padding (w-padding) (h-padding)]);
                [circleCenters, radii, ~] = imfindcircles(searching, uint16([lbradius ubradius]), 'ObjectPolarity', 'dark');
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
        
        function[eyePupil, irisResults, success] = findEye(videoFrame, eyeBox, clusters, debug)
            if nargin < 4
                debug = false;
            end
            success = true;
  
            eyeImage = imcrop(videoFrame, eyeBox);
            [~, m] = size(eyeImage);

            eyeImage = DetectionHelper.toGrayScale(eyeImage);

            [~, ~, pixel_labels] = DetectionHelper.clusterImage(...
                eyeImage, clusters);

            [cluster_images, ~] = DetectionHelper.createClusterImages(clusters, pixel_labels);

            [maxcircle, maxradius, maxcluster] = DetectionHelper.findMaxCircleCluster(cluster_images, 2, 25);
            if maxradius == 0
                success = false;
            end

            if(debug)
                figure; imshow(eyeImage);
                figure; imshow(cluster_images(:, :, maxcluster)); title('MAXCLUSTER');
                viscircles(maxcircle, maxradius, 'EdgeColor', 'r');
            end
            
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
