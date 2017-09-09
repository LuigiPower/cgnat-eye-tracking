per_train = 0.7;
load('generated/samples.mat'); % loads total_samples

nOfSamples = size(total_samples, 1);
nFolds = 100;
logical = zeros(nOfSamples, size(total_samples, 2));

kernel_functions = {'linear', 'quadratic', 'polynomial', 'rbf', 'mlp'};

fp = zeros(size(kernel_functions, 2), 1);
fn = zeros(size(kernel_functions, 2), 1);
cl = zeros(size(kernel_functions, 2), 1);

for  j = 1:size(kernel_functions, 2)
    sumFP = 0;
    sumFN = 0;
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
        classloss = 0;
        
        for cidx = 1:size(v_test, 1)
            if classes(cidx) == test_group(cidx)
                if classes(cidx) == -1
                    % True Negative
                else
                    % True Positive
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
        
        class_l =  class_l + classloss;
    end
    
    fp(j) = sumFP ./ nFolds;
    fn(j) = sumFN ./ nFolds;
    cl(j) = class_l ./ nFolds;
end