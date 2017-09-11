disp('showing files')

%cgpath = '1.Dataset/cg/';
sad = transpose(dir(cgpath));
for file = sad
    if strcmp(file.name, '.') || strcmp(file.name, '..') || strncmp(file.name, '.', 1)
        continue;
    end
    
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

        main
    end
end

%natpath = '1.Dataset/nat/';
sad = transpose(dir(natpath));
for file = sad
    if strcmp(file.name, '.') || strcmp(file.name, '..') || strncmp(file.name, '.', 1)
        continue;
    end
    
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

        main
    end
end