%% COMPUTE_FEATURES
%  Runs the main application to track the eyes in a video and save all the
%  results in the Results folder.

disp('showing files')

%cgpath = '1.Dataset/cg/';
counter = 4;
sad = transpose(dir(cgpath));
for file = sad
    if strcmp(file.name, '.') || strcmp(file.name, '..') || strncmp(file.name, '.', 1)
        continue;
    end
    
    counter = counter + 1;
    
    if file.isdir
        subsad = transpose(dir(strcat(strcat(file.folder, '/'), file.name)));
        
        for subfile = subsad
            if strcmp(subfile.name, '.') || strcmp(subfile.name, '..') || strncmp(subfile.name, '.', 1)
                continue;
            end
            
            path = strcat(subfile.folder, '/');
            absfile = strcat(strcat(subfile.folder, '\'), subfile.name);
            split = strsplit(subfile.name, '.');
            disp(subfile);
            filename = split{1};
            ext = strcat('.', split{2});
            video_class = 'cg';
            video_id = counter;
           
            main
        end
    else
        path = strcat(file.folder, '/');
        absfile = strcat(strcat(file.folder, '\'), file.name);
        split = strsplit(file.name, '.');
        disp(file);
        filename = split{1};
        ext = strcat('.', split{2});
        video_class = 'cg';
        video_id = counter;

        main
    end
end

%natpath = '1.Dataset/nat/';
counter = 4;
sad = transpose(dir(natpath));
for file = sad
    if strcmp(file.name, '.') || strcmp(file.name, '..') || strncmp(file.name, '.', 1)
        continue;
    end
    
    counter = counter + 1;
    
    if file.isdir
        subsad = transpose(dir(strcat(strcat(file.folder, '/'), file.name)));
        
        for subfile = subsad
            if strcmp(subfile.name, '.') || strcmp(subfile.name, '..') || strncmp(subfile.name, '.', 1)
                continue;
            end
            
            path = strcat(subfile.folder, '/');
            absfile = strcat(strcat(subfile.folder, '\'), subfile.name);
            split = strsplit(subfile.name, '.');
            filename = split{1};
            ext = strcat('.', split{2});
            video_class = 'nat';
            video_id = counter;

            main
        end
    else
        path = strcat(file.folder, '/');
        absfile = strcat(strcat(file.folder, '\'), file.name);
        disp(absfile);
        split = strsplit(file.name, '.');
        filename = split{1};
        ext = strcat('.', split{2});
        video_class = 'nat';
        video_id = counter;

        main
    end
end