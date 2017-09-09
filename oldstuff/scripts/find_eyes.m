%% Finds eyes in video frame, returns them as points
faceDetector = vision.CascadeObjectDetector();
visiblePoints = detectMinEigenFeatures(rgb2gray(videoFrame), 'ROI', bbox(2, :));