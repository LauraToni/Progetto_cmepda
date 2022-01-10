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

imageAD={};
imageCTRL={};
for i=2:10 %144
    s=num2str(i);
    file_nameAD=strcat('AD_CTRL/AD_s3/smwc1AD-',s,'.nii');
    imageAD=cat(4,imageAD,niftiread(file_nameAD));
end
for i=2:10 %189
    s=num2str(i);
    file_nameCTRL=strcat('AD_CTRL/CTRL_s3/smwc1CTRL-',s,'.nii');
    imageCTRL=cat(4,imageCTRL,niftiread(file_nameCTRL));
end
figure;
subplot(2,2,1)
imagesc(squeeze(imageAD(:,:,61,3))); colormap gray %la z è fissata, trasversale
subplot(2,2,3)
imagesc(squeeze(imageAD(:,72,:,3))); colormap gray %la y è fissata, coronale
subplot(2,2,4)
imagesc(squeeze(imageAD(61,:,:,3))); colormap gray %la x è fissata, sagittale

%% Create a bounding box

figure;
imageAD_ROI={};

for i=1:10  % we  consider just cubic ROIs
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

