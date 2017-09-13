%% CLASSIFY
%  Runs classification of the outputValues in the workspace using the SVM
%  specified by the .mat file chosen by the user
%  NOTE: outputValues must be in the workspace

[fname, path] = uigetfile('*.mat', 'Select a .mat file containing the trained SVM');

%load('3.Results/trained.mat'); % loads trained SVM
load(strcat(path, fname));

outputValues
v_test = outputValues;
to_remove = [];
for i = 1:size(outputValues, 1)
    if outputValues(i, 1) == 0
        to_remove = [ i; to_remove ]
    end
end
v_test(to_remove, :) = [];
test_group = v_test(:, 1);
v_test(:, 1:3) = [];

[classes] = svmclassify(SVMTrain, v_test);

%  1 CG
% -1 NAT
cg = 0;
nat = 0;
for i = 1:size(classes, 1)
    if classes(i) == 1
        cg = cg + 1;
    else
        nat = nat + 1;
    end
end

classes
cg
nat

if cg > nat
    h = msgbox('Video is cg');
else
    h = msgbox('Video is nat');
end