function img = load_image()
% num is the num of image with 

path = 'E:\AutomaticRedChannelRestoration\AutomaticRedChannelRestoration\image';
filename = dir(fullfile(hazyimgs_path,'*.png'));
b = length(filename);
for j =1:b
    fname = strcat(hazyimgs_path, '/', filename(j).name);
    img =imread(fname);
