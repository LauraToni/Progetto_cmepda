%% Implement preprocessing NifTi images cutting borders and saving them in a new folder

%The objective is to upload and read NifTi files from the directory
%"AD_CTRL": 
%the dyrectory contains to other dyrectories containing to different
%dataset

close all
clear
clc
%% Read csv file containing metadata
%Read csv file containing metadata and display it
%Add labels column
%Save the modified table in a new file.csv

filename='AD_CTRL/AD_CTRL_metadata.csv';

metadata=readtable(filename, 'ReadRowNames', true);

metadata.Properties


features=metadata(:,:)
metadata.ClassLabel=categorical(metadata.DXGROUP)

for i=1:333
    if metadata.ClassLabel(i)=='AD'
        metadata.Labels(i)=1;
    else metadata.Labels(i)=-1;
    end
end

summary(metadata)  

metadata = removevars(metadata,{'ClassLabel'});

writetable(metadata,'AD_CTRL_metadata_labels.csv','WriteRowNames',true);

%% Read NifTi files

%Upload a NifTi image

dataset_path='AD_CTRL/';
file_name='CTRL_s3/smwc1CTRL-167.nii';
file_path=(strcat(dataset_path,file_name));

V = niftiread(file_path);

%Display a slice of the image

figure; imagesc(V(:,:, 30)); colormap gray






