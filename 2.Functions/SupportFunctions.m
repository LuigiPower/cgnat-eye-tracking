classdef SupportFunctions
    methods (Static)
        
        %% removeinfnan
        % remove inf or nan from array
        function[res] = removeinfnan(arr)
            res = arr;
            res(res == inf | isnan(res)) = [];
        end
        
        %% points2bbox
        % get bounding rectangle of a list of 4 points
        function[bbox] = points2bbox(points)
            xmin = min(points(:,1));
            xmax = max(points(:,1));
            ymin = min(points(:,2));
            ymax = max(points(:,2));
            bbox = [xmin ymin (xmax - xmin) (ymax - ymin)];
        end
        
        %% orderDescByArea
        % Order Boxes by area descending
        function[maxarea, maxindex] = orderDescByArea(bboxes)
            [n, ~] = size(bboxes);
            bboxes
            prods = zeros(n);
            for i = 1:n
                prods(i) = prod(bboxes(i, 3:4));
            end
            [maxarea, maxindex] = max(prods);
            prods
        end
        
        %% orderAscByArea
        % Order Boxes by area descending
        function[minarea, minindex] = orderAscByArea(bboxes)
            [n, ~] = size(bboxes);
            prods = zeros(n);
            for i = 1:n
                prods(i) = prod(bboxes(i, 3:4));
            end
            [minarea, minindex] = min(prods);
        end
        
        %% getMaxIntersect
        % Order by overlap ratio
        function[maxintersect, i, j] = getMaxIntersect(bboxes1, bboxes2)
            [n1, ~] = size(bboxes1);
            [n2, ~] = size(bboxes2);
            intersects = zeros(n1, n2);
            for i = 1:n1
                for j = 1:n2
                    intersects(i, j) = bboxOverlapRatio(bboxes1(i, :), bboxes2(j, :));
                end
            end
            [intersects2, index] = max(intersects);
            [maxintersect, j] = max(intersects2);
            i = index(j);
        end
        
        %% removeNonIntersecting
        % Removes all non intersecting bounding boxes from first argument
        function[boxes] = removeNonIntersecting(bboxes1, bboxes2, threshold)
            [n1, m1] = size(bboxes1);
            [n2, ~] = size(bboxes2);
            boxes = int16.empty(0, m1);
            for i = 1:n1
                for j = 1:n2
                    intersection = bboxOverlapRatio(bboxes1(i, :), bboxes2(j, :));
                    if(intersection > threshold)
                        boxes = [boxes;bboxes1(i, :)];
                        %boxes(top, :) = bboxes1(i, :);
                    end
                end
            end
        end
        
        %% getLeftMost
        % Get left most bounding box
        function[box] = getLeftMost(bboxes)
            [n1, ~] = size(bboxes);
            if n1 == 0
                box = [];
            else
                box = bboxes(1, :);
                minimum = box(1, 1);
                for i = 1:n1
                    if(bboxes(i, 1) < minimum)
                        box = bboxes(i, :);
                        minimum = bboxes(i, 1);
                    end
                end
            end
        end
        
        %% getRightMost
        % Get right most bounding box
        function[box] = getRightMost(bboxes)
            [n1, ~] = size(bboxes);
            if n1 == 0
                box = [];
            else
                box = bboxes(1, :);
                maximum = box(1, 1);
                for i = 1:n1
                    bboxes(i, 1)
                    if(bboxes(i, 1) > maximum)
                        box = bboxes(i, :);
                        maximum = bboxes(i, 1);
                    end
                end
            end
        end
        
        %% getCenter
        % get the center point of the bounding box
        function[point] = getCenter(bbox)
            point = [double((bbox(1, 1) + bbox(1, 3) / 2)) double((bbox(1, 2) + bbox(1, 4) / 2))];
        end
    end
end
