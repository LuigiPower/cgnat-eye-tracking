[fname, path] = uigetfile('*.mat', 'Select a .mat file containing the Samples');

%load('1.Dataset/samples.mat'); % loads total_samples
load(strcat(path, fname));

nOfSamples = size(total_samples, 1);
[v_train, idx] = datasample(total_samples, nOfSamples, 'Replace', false);

% need to splice total_samples rows for train and test, because
% columns 1-4 included don't matter to the SVM
group = v_train(:, 1);
v_train(:, 1:4) = [];

SVMTrain = svmtrain(v_train, group, 'kernel_function', 'rbf');

[filename, pathname] = uiputfile('trained.mat');

%save('3.Results/trained.mat', 'SVMTrain');
save(strcat(pathname, filename), 'SVMTrain');