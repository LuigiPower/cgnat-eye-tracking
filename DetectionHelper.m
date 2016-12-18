classdef DetectionHelper
    methods (Static)
        %% toGrayScale
        function[image] = toGrayScale(image)
            % toGrayScale
            % Creates 3 channel gray scale
            %   @param image image to convert
            
            imageGS = rgb2gray(image);
            image(:, :, 1) = imageGS;
            image(:, :, 2) = imageGS;
            image(:, :, 3) = imageGS;
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
                                                       'sqEuclidean', 'Replicates', 3);

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
                padding = h/4;
                searching = imcrop(cluster_images(:, :, i), [padding padding (w-padding) (h-padding)]);
                [circleCenters, radii, ~] = imfindcircles(searching, uint16([lbradius ubradius]));
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
        
        function[eyePupil, irisResults] = findEye(videoFrame, eyeBox, clusters)
            eyeImage = imcrop(videoFrame, eyeBox);
            [~, m] = size(eyeImage);

            eyeImage = DetectionHelper.toGrayScale(eyeImage);

            [~, ~, pixel_labels] = DetectionHelper.clusterImage(...
                eyeImage, clusters);

            [cluster_images, ~] = DetectionHelper.createClusterImages(clusters, pixel_labels);

            [maxcircle, maxradius, maxcluster] = DetectionHelper.findMaxCircleCluster(cluster_images, 1, m/2 - 1);

            %figure; imshow(cluster_images(:, :, maxcluster)); title('MAXCLUSTER');
            %viscircles(maxcircle, maxradius, 'EdgeColor', 'r');

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
