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

%figure; imagesc(squeeze(V(:,30,:))); colormap gray

%Upload file Nifti in a 4D matrix where the first, second and third
%dimension are the image voxels and the forth dimension is the
%concatenation direction

imageAD=[];
imageCTRL=[];
disp('Caricamento immagini AD')
for i=1:144 %144
    disp(i)
    s=num2str(i);
    file_nameAD=strcat('AD_CTRL/AD_s3/smwc1AD-',s,'.nii');
    imageAD=cat(4,imageAD,niftiread(file_nameAD));
end

disp('Caricamento immagini CTRL')
for i=1:189 %189
    disp(i)
    s=num2str(i);
    file_nameCTRL=strcat('AD_CTRL/CTRL_s3/smwc1CTRL-',s,'.nii');
    imageCTRL=cat(4,imageCTRL,niftiread(file_nameCTRL));
end

%Visualize the largest slice of an image for each dimension
%to find the optimal ROI to minimize the black border

figure;
subplot(2,2,1)
imagesc(squeeze(imageAD(:,:,61,3))); colormap gray %la z è fissata, trasversale
subplot(2,2,3)
imagesc(squeeze(imageAD(:,72,:,3))); colormap gray %la y è fissata, coronale
subplot(2,2,4)
imagesc(squeeze(imageAD(61,:,:,3))); colormap gray %la x è fissata, sagittale

%% Create a bounding box

%Bounging box AD images

imageAD_ROI=[];

for i=1:144  % we  consider just cubic ROIs
    disp(i)
    P1a=7;
    P1b=114;
    P2a=6;
    P2b=140;
    P3a=1;
    P3b=109;
    ROI_P=imageAD(P1a:P1b,P2a:P2b,P3a:P3b,i);
    ROI_P=squeeze(ROI_P);
    imageAD_ROI=cat(4,imageAD_ROI,ROI_P);
end

%Bounging box CTRL images

imageCTRL_ROI=[];

for i=1:189  % we  consider just cubic ROIs
    disp(i)
    P1a=7; %7
    P1b=114; %114
    P2a=6; %6
    P2b=140; %140
    P3a=1; %1
    P3b=109; %109
    ROI_P=imageCTRL(P1a:P1b,P2a:P2b,P3a:P3b,i);
    ROI_P=squeeze(ROI_P);
    imageCTRL_ROI=cat(4,imageCTRL_ROI,ROI_P);
end

%Visualize imageAD_ROI
%visualize the largest slice of one image, in each dimension.

figure;
subplot(2,2,1)
imagesc(squeeze(imageAD_ROI(:,:,61,8))); colormap gray %la z è fissata, trasversale
subplot(2,2,3)
imagesc(squeeze(imageAD_ROI(:,72,:,8))); colormap gray %la y è fissata, coronale
subplot(2,2,4)
imagesc(squeeze(imageAD_ROI(61,:,:,8))); colormap gray %la x è fissata, sagittale
title("Tagliata AD")


%Visualize imageCTRL_ROI
%visualize the largest slice of one image, in each dimension.

figure;
subplot(2,2,1)
imagesc(squeeze(imageCTRL_ROI(:,:,61,8))); colormap gray %la z è fissata, trasversale
subplot(2,2,3)
imagesc(squeeze(imageCTRL_ROI(:,72,:,8))); colormap gray %la y è fissata, coronale
subplot(2,2,4)
imagesc(squeeze(imageCTRL_ROI(61,:,:,8))); colormap gray %la x è fissata, sagittale
title("Tagliata CTRL")

%% Output 
%create two folders in 'AD_CTRL/' for each ROI set 
%Save modified images in NifTi format in the two folders separating AD and
%CTRL

%Defining the dir and the names of the output files

fileID='AD_CTRL/AD_CTRL_metadata.csvAD_CTRL_metadata.csv';

[filepath,name,ext] = fileparts(fileID);
fileOUTpath_AD=fullfile(filepath,'AD_ROI/');
fileOUTpath_CTRL=fullfile(filepath,'CTRL_ROI/');

if ~exist(fileOUTpath_AD, 'dir')
    mkdir(fileOUTpath_AD);
end

if ~exist(fileOUTpath_CTRL, 'dir')
    mkdir(fileOUTpath_CTRL);
end

%Saving the AD images

disp('Writing the output AD files');

for i=1:144 %144
    disp(i)
    s=num2str(i);
    fileIDout_AD=strcat(fileOUTpath_AD,'smwc1AD-',s,'_ROI','.nii');
    niftiwrite(imageAD_ROI(:,:,:,i),fileIDout_AD);
end

disp('... done!');

%Saving the CTRL images

disp('Writing the output CTRL files');

for i=1:189 %189
    disp(i)
    s=num2str(i);
    fileIDout_CTRL=strcat(fileOUTpath_CTRL,'smwc1CTRL-',s,'_ROI','.nii');
    niftiwrite(imageCTRL_ROI(:,:,:,i),fileIDout_CTRL);
end

disp('... done!');





