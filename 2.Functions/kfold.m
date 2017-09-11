per_train = 0.7;

[fname, path] = uigetfile('*.mat', 'Select a .mat file containing the Samples');

%load('1.Dataset/samples.mat'); % loads total_samples
load(strcat(path, fname));

nOfSamples = size(total_samples, 1);
nFolds = 20;
logical = zeros(nOfSamples, size(total_samples, 2));

kernel_functions = {'linear', 'quadratic', 'polynomial', 'rbf', 'mlp'};

fp = zeros(size(kernel_functions, 2), 1);
fn = zeros(size(kernel_functions, 2), 1);
tp = zeros(size(kernel_functions, 2), 1);
tn = zeros(size(kernel_functions, 2), 1);
precision = zeros(size(kernel_functions, 2), 1);
recall = zeros(size(kernel_functions, 2), 1);
f1 = zeros(size(kernel_functions, 2), 1);
cl = zeros(size(kernel_functions, 2), 1);

for  j = 1:size(kernel_functions, 2)
    sumFP = 0;
    sumFN = 0;
    sumTP = 0;
    sumTN = 0;
    class_l = 0;
    
    for i = 1:nFolds
        [v_train, idx] = datasample(total_samples, uint16(nOfSamples * per_train), 'Replace', false);
        v_test = total_samples;
        v_test(idx, :) = [];
        
        % need to splice total_samples rows for train and test, because
        % columns 1-4 included don't matter to the SVM
        group = v_train(:, 1);
        test_group = v_test(:, 1);
        v_train(:, 1:4) = [];
        v_test(:, 1:4) = [];
        
        %matrix_train = compute_params(v_train);
        SVMTrain = svmtrain(v_train, group, 'kernel_function', kernel_functions{j});
        
        [classes] = svmclassify(SVMTrain, v_test);
        
        FN = 0;
        FP = 0;
        TP = 0;
        TN = 0;
        classloss = 0;
        
        for cidx = 1:size(v_test, 1)
            if classes(cidx) == test_group(cidx)
                if classes(cidx) == -1
                    % True Negative
                    TP = TP + 1;
                else
                    % True Positive
                    TN = TN + 1;
                end
            else
                classloss = classloss + 1;
                if classes(cidx) == -1
                    % False Negative
                    FN = FN + 1;
                else
                    % False Positive
                    FP = FP + 1;
                end
            end
        end
        
        sumFP = sumFP + FP;
        sumFN = sumFN + FN;
        sumTP = sumTP + TP;
        sumTN = sumTN + TN;
        
        class_l =  class_l + classloss;
    end
    
    fp(j) = sumFP ./ nFolds;
    fn(j) = sumFN ./ nFolds;
    tp(j) = sumTP ./ nFolds;
    tn(j) = sumTN ./ nFolds;
    precision(j) = tp(j) / (tp(j) + fp(j));
    recall(j) = tp(j) / (fn(j) + tp(j));
    f1 = (2 * precision(j) * recall(j)) / (precision(j) + recall(j));
    cl(j) = class_l ./ nFolds;
end