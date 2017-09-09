disp('showing files')

path = 'videos/cg/';

sad = transpose(dir(path));
for file = sad
    if strcmp(file.name, '.') || strcmp(file.name, '..') || strncmp(file.name, '.', 1)
        continue;
    end
    
    absfile = strcat(strcat(file.folder, '\'), file.name);
    split = strsplit(file.name, '.');
    filename = split{1};
    ext = strcat('.', split{2});
    disp(file);
    
    track_eyes
end

path = 'videos/natural/';
sad = transpose(dir(path));
for file = sad
    if strcmp(file.name, '.') || strcmp(file.name, '..') || strncmp(file.name, '.', 1)
        continue;
    end
    
    absfile = strcat(strcat(file.folder, '\'), file.name);
    disp(absfile);
    split = strsplit(file.name, '.');
    filename = split{1};
    ext = strcat('.', split{2});
    
    track_eyes
end