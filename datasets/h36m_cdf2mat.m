% This script is the pre-process for h36m dataset processing.

% H36M dataset save the pose sequence as cdf file, which require complex
% dependencies to open in python.  So we use matlab to transform all cdf
% files to mat files before processing.

% Before running the script, please set the h36m_path (line 13) to your 
% H36M unzip path.

clc
clear all

h36m_path = '/media/hao/DATA/H36M/';

h36m_list = fileread('./h36m_list.txt');
h36m_list = strsplit(h36m_list, '\r\n');
num = str2num(h36m_list{1});

for i=1:num
    src_file = strcat(h36m_path, h36m_list{i*3+1}(1:end-3), 'cdf');
    tgt_file = strcat(h36m_path, h36m_list{i*3+1});
    content = cdfread(src_file);
    pose = content{1};
    save(tgt_file, 'pose')
end
disp('Done')