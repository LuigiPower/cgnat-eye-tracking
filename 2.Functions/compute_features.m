disp('showing files')

%cgpath = '1.Dataset/cg/';
sad = transpose(dir(cgpath));
for file = sad
    if strcmp(file.name, '.') || strcmp(file.name, '..') || strncmp(file.name, '.', 1)
        continue;
    end
    
    absfile = strcat(strcat(file.folder, '\'), file.name);
    split = strsplit(file.name, '.');
    filename = split{1};
    ext = strcat('.', split{2});
    video_class = 'cg';
    
    track_eyes
end

%natpath = '1.Dataset/nat/';
sad = transpose(dir(natpath));
for file = sad
    if strcmp(file.name, '.') || strcmp(file.name, '..') || strncmp(file.name, '.', 1)
        continue;
    end
    
    absfile = strcat(strcat(file.folder, '\'), file.name);
    disp(absfile);
    split = strsplit(file.name, '.');
    filename = split{1};
    ext = strcat('.', split{2});
    video_class = 'nat';
    
    track_eyes
end