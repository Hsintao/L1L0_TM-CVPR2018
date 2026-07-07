clear;clc;
script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir)
    script_dir = pwd;
end
addpath(script_dir);

%% Parameters
lambda1 = 0.3;  
lambda2 = lambda1*0.01;
lambda3 = 0.1;

%% read files
input_path = fullfile(script_dir,'..','inputs','1.hdr');
hdr = hdrimread(input_path);
[hei,wid,channel] = size(hdr);

tic;
%% transformation
hdr_h = rgb2hsv(hdr);
hdr_l = hdr_h(:,:,3);
hdr_l = log(hdr_l+0.0001);
hdr_l = nor(hdr_l);

%%  decomposition
[D1,D2,B2] = Layer_decomp(hdr_l,lambda1,lambda2,lambda3);

%% Scaling
sigma_D1 = max(D1(:));
D1s = R_func(D1,0,sigma_D1,0.8,1);
% sigma_D2 = max(D2(:));
% D2s = R_func(D2,0,sigma_D2,0.9,1);
B2_n= compress(B2,2.2,1);
hdr_lnn = 0.8*B2_n + D2 + 1.2*D1s;

%% postprocessing
hdr_lnn = nor(clampp(hdr_lnn,0.005,0.995));
out_rgb = hsv2rgb((cat(3,hdr_h(:,:,1),hdr_h(:,:,2)*0.6,hdr_lnn)));
toc;

output_dir = fullfile(script_dir,'..','results');
if ~exist(output_dir,'dir')
    mkdir(output_dir);
end
imwrite(out_rgb, fullfile(output_dir,'demo_matlab.png'));
figure,imshow(out_rgb)



