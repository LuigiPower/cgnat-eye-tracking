%% FIGURES.m
%
%  AUTHOR: Mattia BONOMI - MMLAB 2016
%
%  Used to import figures from results and convert them in a proper way.

fpath = './Figures';


%cgpath = '1.Dataset/cg/';
sad = transpose(dir(fpath));
for file = sad
    if strcmp(file.name, '.') || strcmp(file.name, '..') || strncmp(file.name, '.', 1)
        continue;
    end
    
    path = strcat(file.folder, '/');
    absfile = strcat(strcat(file.folder, '/'), file.name);
    split = strsplit(file.name, '.');
    disp(file);
    filename = split{1};
    ext = strcat('.', split{2});
    
    h = openfig(absfile,'invisible');
    saveas(h, [path filename], 'jpg');
end