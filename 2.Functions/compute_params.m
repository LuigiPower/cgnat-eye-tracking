%cgpath = 'generated/cg/';
%natpath = 'generated/nat/';

total_samples = zeros(0, 20); % 20 are 18 features + sample validity + label

sad = transpose(dir(cgpath));
for file = sad
    if strcmp(file.name, '.') || strcmp(file.name, '..') || strncmp(file.name, '.', 1)
        continue;
    end
    
    absfolder = strcat(strcat(file.folder, '\'), file.name);
    disp(absfolder);
    
    load(strcat(absfolder, '\features.mat'));
    
    for index = 1:size(outputValues, 1)
        if outputValues(index, 1) == 0
            continue; % First column of sample set to 0 means it is not a real sample
        end
        
        valid = 1;
        for validx = 1:size(outputValues, 2)
            value = outputValues(index, validx);
            if isnan(value) || value == inf
                valid = 0;
                break;
            end
        end
        
        if valid == 1
            total_samples = [total_samples; [1 outputValues(index, :)]]; % 1 means CG
        end
    end
end


sad = transpose(dir(natpath));
for file = sad
    if strcmp(file.name, '.') || strcmp(file.name, '..') || strncmp(file.name, '.', 1)
        continue;
    end
    
    absfolder = strcat(strcat(file.folder, '\'), file.name);
    disp(absfolder);
    
    load(strcat(absfolder, '\features.mat'));
    
    for index = 1:size(outputValues, 1)
        if outputValues(index, 1) == 0
            continue; % First column of sample set to 0 means it is not a real sample
        end
        
        valid = 1;
        for validx = 1:size(outputValues, 2)
            value = outputValues(index, validx);
            if isnan(value) || value == inf
                valid = 0;
                break;
            end
        end
        
        if valid == 1
            total_samples = [total_samples; [-1 outputValues(index, :)]]; % -1 means NAT
        end
    end
end


[filename, pathname] = uiputfile('samples.mat');

%save('generated/samples.mat', 'total_samples');
save(strcat(pathname, filename), 'total_samples');

