%% Implement preprocessing NifTi images cutting borders and saving them in a new folder

% - Upload and read NifTi files from the directory;
% - Define a ROI containing only the hyppocampus and save them in a folder;
% - Define a ROI containing a region that doesn't contain the hyppocampus and
%save them in a folder;
% - Print two images hilighting the ROIs considered.

close all
clear
clc
%% Read csv file containing metadata
% - Read csv file containing metadata and display it;
% - Add labels column;
% - Save the modified table in a new file.csv.

filename='AD_CTRL/AD_CTRL_metadata.csv';

metadata=readtable(filename, 'ReadRowNames', true);

metadata.Properties;
features=metadata(:,:);
metadata.ClassLabel=categorical(metadata.DXGROUP);

% Add labels column
for i=1:333
    if metadata.ClassLabel(i)=='AD'
        metadata.Labels(i)=1;
    else metadata.Labels(i)=0;
    end
end

summary(metadata)  

% Save the new metadata
metadata = removevars(metadata,{'ClassLabel'});
writetable(metadata,'AD_CTRL_metadata_labels.csv','WriteRowNames',true);

%% Read NifTi files

%Upload a NifTi image

dataset_path='AD_CTRL/';
file_name='CTRL_s3/smwc1CTRL-167.nii';
file_path=(strcat(dataset_path,file_name));

V = niftiread(file_path);

% Display a slice of the image
%figure; imagesc(squeeze(V(:,30,:))); colormap gray

% Upload file Nifti in a 4D matrix where the first, second and third
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
imagesc(squeeze(imageAD(:,:,50,3))); colormap gray %la z è fissata, trasversale
title(strcat('TRASVERSE view - slice= ', num2str(50)));
subplot(2,2,3)
imagesc(squeeze(imageAD(:,72,:,3))); colormap gray %la y è fissata, coronale
title(strcat('CORONAL view - slice= ', num2str(72)));
subplot(2,2,4)
imagesc(squeeze(imageAD(61,:,:,3))); colormap gray %la x è fissata, sagittale
title(strcat('SAGITTAL view - slice= ', num2str(61)));

%% Create a bounding box

% Bounding box AD images

imageAD_ROI=[];

for i=1:144  % we consider just cubic ROIs 144
    disp(i)
    P1a=11;
    P1b=109;
    P2a=12;
    P2b=138;
    P3a=24;
    P3b=110;
    ROI_P=imageAD(P1a:P1b,P2a:P2b,P3a:P3b,i);
    ROI_P=squeeze(ROI_P);
    imageAD_ROI=cat(4,imageAD_ROI,ROI_P);
end

% Bounding box CTRL images

imageCTRL_ROI=[];

for i=1:189  % we consider just cubic ROIs 189
    disp(i)
    P1a=11; %7
    P1b=109; %114
    P2a=12; %6
    P2b=138; %140
    P3a=24; %1
    P3b=110; %109
    ROI_P=imageCTRL(P1a:P1b,P2a:P2b,P3a:P3b,i);
    ROI_P=squeeze(ROI_P);
    imageCTRL_ROI=cat(4,imageCTRL_ROI,ROI_P);
end

% Visualize imageAD_ROI
% visualize the largest slice of one image, in each dimension.

figure;
subplot(2,2,1)
imagesc(squeeze(imageAD_ROI(:,:,3,3))); colormap gray %la z è fissata, trasversale
title(strcat('ROI AD- TRASVERSE view - slice= ', num2str(3)));
subplot(2,2,3)
imagesc(squeeze(imageAD_ROI(:,3,:,3))); colormap gray %la y è fissata, coronale
title(strcat('ROI AD - CORONAL view - slice= ', num2str(3)));
subplot(2,2,4)
imagesc(squeeze(imageAD_ROI(3,:,:,3))); colormap gray %la x è fissata, sagittale
title(strcat('ROI AD- SAGITTAL view - slice= ', num2str(3)));



% Visualize imageCTRL_ROI
% visualize the largest slice of one image, in each dimension.

figure;
subplot(2,2,1)
imagesc(squeeze(imageCTRL_ROI(:,:,3,3))); colormap gray %la z è fissata, trasversale
title(strcat('ROI CTRL- TRASVERSE view - slice= ', num2str(3)));
subplot(2,2,3)
imagesc(squeeze(imageCTRL_ROI(:,3,:,3))); colormap gray %la y è fissata, coronale
title(strcat('ROI CTRL- CORONAL view - slice= ', num2str(3)));
subplot(2,2,4)
imagesc(squeeze(imageCTRL_ROI(3,:,:,3))); colormap gray %la x è fissata, sagittale
title(strcat('ROI CTRL- SAGITTAL view - slice= ', num2str(3)));

%% Create a bounding box of a region without the hyppocampus

% Bounging box AD images

imageAD_ROI_VOID=[];

for i=1:144  % we consider just cubic ROIs 144
    disp(i)
    P1av=35;
    P1bv=84;
    P2av=85;
    P2bv=134;
    P3av=55;
    P3bv=104;
    ROI_P_VOID=imageAD(P1av:P1bv,P2av:P2bv,P3av:P3bv,i);
    ROI_P_VOID=squeeze(ROI_P_VOID);
    imageAD_ROI_VOID=cat(4,imageAD_ROI_VOID,ROI_P_VOID);
end

% Bounging box CTRL images

imageCTRL_ROI_VOID=[];

for i=1:189  % we  consider just cubic ROIs 189
    disp(i)
    P1av=35; %7
    P1bv=84; %114
    P2av=85; %6
    P2bv=134; %140
    P3av=55; %1
    P3bv=104; %109
    ROI_P_VOID=imageCTRL(P1av:P1bv,P2av:P2bv,P3av:P3bv,i);
    ROI_P_VOID=squeeze(ROI_P_VOID);
    imageCTRL_ROI_VOID=cat(4,imageCTRL_ROI_VOID,ROI_P_VOID);
end

% Visualize imageAD_ROI_VOID
% Visualize the largest slice of one image, in each dimension.

figure;
subplot(2,2,1)
imagesc(squeeze(imageAD_ROI_VOID(:,:,25,3))); colormap gray %la z è fissata, trasversale
title(strcat('ROI AD VOID - TRASVERSE view - slice= ', num2str(25)));
subplot(2,2,3)
imagesc(squeeze(imageAD_ROI_VOID(:,25,:,3))); colormap gray %la y è fissata, coronale
title(strcat('ROI AD VOID - CORONAL view - slice= ', num2str(25)));
subplot(2,2,4)
imagesc(squeeze(imageAD_ROI_VOID(25,:,:,3))); colormap gray %la x è fissata, sagittale
title(strcat('ROI AD VOID - SAGITTAL view - slice= ', num2str(25)));


% Visualize imageCTRL_ROI_VOID
% Visualize the largest slice of one image, in each dimension.

figure;
subplot(2,2,1)
imagesc(squeeze(imageCTRL_ROI_VOID(:,:,25,3))); colormap gray %la z è fissata, trasversale
title(strcat('ROI CTRL VOID - TRASVERSE view - slice= ', num2str(25)));
subplot(2,2,3)
imagesc(squeeze(imageCTRL_ROI_VOID(:,25,:,3))); colormap gray %la y è fissata, coronale
title(strcat('ROI CTRL VOID - CORONAL view - slice= ', num2str(25)));
subplot(2,2,4)
imagesc(squeeze(imageCTRL_ROI_VOID(25,:,:,3))); colormap gray %la x è fissata, sagittale
title(strcat('ROI CTRL VOID - SAGITTAL view - slice= ', num2str(25)));

%% Create a bounding box for the hyppocampus

% Bounding box AD images

imageAD_ROI_TH=[];

for i=1:144  % we consider just cubic ROIs 144
    disp(i)
    P1at=36;
    P1bt=85;
    P2at=51;
    P2bt=100;
    P3at=26;
    P3bt=75;
    ROI_P_TH=imageAD(P1at:P1bt,P2at:P2bt,P3at:P3bt,i);
    ROI_P_TH=squeeze(ROI_P_TH);
    imageAD_ROI_TH=cat(4,imageAD_ROI_TH,ROI_P_TH);
end

% Bounding box CTRL images

imageCTRL_ROI_TH=[];

for i=1:189  % we  consider just cubic ROIs 189
    disp(i)
    P1at=36; 
    P1bt=85; 
    P2at=51; %58
    P2bt=100; %108
    P3at=26; 
    P3bt=75; 
    ROI_P_TH=imageCTRL(P1at:P1bt,P2at:P2bt,P3at:P3bt,i);
    ROI_P_TH=squeeze(ROI_P_TH);
    imageCTRL_ROI_TH=cat(4,imageCTRL_ROI_TH,ROI_P_TH);
end

% Visualize imageAD_ROI_TH
% Visualize the largest slice of one image, in each dimension.

figure;
subplot(2,2,1)
imagesc(squeeze(imageAD_ROI_TH(:,:,25,3))); colormap gray %la z è fissata, trasversale
title(strcat('ROI AD TH - TRASVERSE view - slice= ', num2str(25)));
subplot(2,2,3)
imagesc(squeeze(imageAD_ROI_TH(:,25,:,3))); colormap gray %la y è fissata, coronale
title(strcat('ROI AD TH - CORONAL view - slice= ', num2str(25)));
subplot(2,2,4)
imagesc(squeeze(imageAD_ROI_TH(25,:,:,3))); colormap gray %la x è fissata, sagittale
title(strcat('ROI AD TH - SAGITTAL view - slice= ', num2str(25)));



% Visualize imageCTRL_ROI_TH
% Visualize the largest slice of one image, in each dimension.

figure;
subplot(2,2,1)
imagesc(squeeze(imageCTRL_ROI_TH(:,:,25,3))); colormap gray %la z è fissata, trasversale
title(strcat('ROI CTRL TH - TRASVERSE view - slice= ', num2str(25)));
subplot(2,2,3)
imagesc(squeeze(imageCTRL_ROI_TH(:,25,:,3))); colormap gray %la y è fissata, coronale
title(strcat('ROI CTRL TH - CORONAL view - slice= ', num2str(25)));
subplot(2,2,4)
imagesc(squeeze(imageCTRL_ROI_TH(25,:,:,3))); colormap gray %la x è fissata, sagittale
title(strcat('ROI CTRL TH - SAGITTAL view - slice= ', num2str(25)));

%% Output 
% Create two folders in 'AD_CTRL/' for each ROI set 
% Save modified images in NifTi format in the two folders separating AD and
%CTRL

% Defining the dir and the names of the output files

fileID='AD_CTRL/AD_CTRL_metadata.csv';

[filepath,name,ext] = fileparts(fileID);
fileOUTpath_AD_VOID=fullfile(filepath,'AD_ROI_VOID/');
fileOUTpath_AD_TH=fullfile(filepath,'AD_ROI_TH/');
fileOUTpath_CTRL_VOID=fullfile(filepath,'CTRL_ROI_VOID/');
fileOUTpath_CTRL_TH=fullfile(filepath,'CTRL_ROI_TH/');

if ~exist(fileOUTpath_AD_VOID, 'dir')
    mkdir(fileOUTpath_AD_VOID);
end

if ~exist(fileOUTpath_CTRL_VOID, 'dir')
    mkdir(fileOUTpath_CTRL_VOID);
end

if ~exist(fileOUTpath_AD_TH, 'dir')
    mkdir(fileOUTpath_AD_TH);
end

if ~exist(fileOUTpath_CTRL_TH, 'dir')
    mkdir(fileOUTpath_CTRL_TH);
end

% Saving the AD images

disp('Writing the output AD files');

for i=1:144 %144
    disp(i)
    s=num2str(i);
    fileIDout_AD_VOID=strcat(fileOUTpath_AD_VOID,'smwc1AD-',s,'_ROI_VOID','.nii');
    niftiwrite(imageAD_ROI_VOID(:,:,:,i),fileIDout_AD_VOID);
    fileIDout_AD_TH=strcat(fileOUTpath_AD_TH,'smwc1AD-',s,'_ROI_TH','.nii');
    niftiwrite(imageAD_ROI_TH(:,:,:,i),fileIDout_AD_TH);
end

disp('... done!');

% Saving the CTRL images

disp('Writing the output CTRL files');

for i=1:189 %189
    disp(i)
    s=num2str(i);
    fileIDout_CTRL_VOID=strcat(fileOUTpath_CTRL_VOID,'smwc1CTRL-',s,'_ROI_VOID','.nii');
    niftiwrite(imageCTRL_ROI_VOID(:,:,:,i),fileIDout_CTRL_VOID);
    fileIDout_CTRL_TH=strcat(fileOUTpath_CTRL_TH,'smwc1CTRL-',s,'_ROI_TH','.nii');
    niftiwrite(imageCTRL_ROI_TH(:,:,:,i),fileIDout_CTRL_TH);
end

disp('... done!');

%% Display a rectangle enclosing the hyppocampus

ROI_H=zeros(size(imageAD));
ROI_H(P1at:P1bt,P2at:P2bt,P3at:P3bt)=imageAD(P1at:P1bt,P2at:P2bt,P3at:P3bt,4);
maximum=max(max(ROI_H(:,:,P3at)));
imageAD_RECH=imageAD(:,:,:,4);

% we display a rectangle around the ROI
imageAD_RECH(P1at:P1bt,P2at,P3bt)=maximum;
imageAD_RECH(P1at:P1bt,P2bt,P3bt)=maximum;
imageAD_RECH(P1at:P1bt,P2bt,P3at)=maximum;
imageAD_RECH(P1at:P1bt,P2at,P3at)=maximum;

imageAD_RECH(P1at,P2at:P2bt,P3bt)=maximum;
imageAD_RECH(P1bt,P2at:P2bt,P3bt)=maximum;
imageAD_RECH(P1at,P2at:P2bt,P3at)=maximum;
imageAD_RECH(P1bt,P2at:P2bt,P3at)=maximum;

imageAD_RECH(P1at,P2at,P3at:P3bt)=maximum;
imageAD_RECH(P1bt,P2at,P3at:P3bt)=maximum;
imageAD_RECH(P1at,P2bt,P3at:P3bt)=maximum;
imageAD_RECH(P1bt,P2bt,P3at:P3bt)=maximum;

figure;
subplot(2,2,1)
imagesc(squeeze(imageAD_RECH(:,:,P3bt))); colormap gray %la z è fissata, trasversale
title(strcat('Hippocampus region - TRASVERSE view - slice= ', num2str(P3at)));
subplot(2,2,3)
imagesc(squeeze(imageAD_RECH(:,P2at,:))); colormap gray %la y è fissata, coronale
title(strcat('Hippocampus region - CORONAL view - slice= ', num2str(P2at)));
subplot(2,2,4)
imagesc(squeeze(imageAD_RECH(P1at,:,:))); colormap gray %la x è fissata, sagittale
title(strcat('Hippocampus region - SAGITTAL view - slice= ', num2str(P1at)));

%% Display a rectangle enclosing the void region

ROI_V=zeros(size(imageAD));
ROI_V(P1av:P1bv,P2av:P2bv,P3av:P3bv)=imageAD(P1av:P1bv,P2av:P2bv,P3av:P3bv,4);
maximum=max(max(ROI_V(:,:,P3av)));
imageAD_RECV=imageAD(:,:,:,4);

% we display a rectangle around the ROI
imageAD_RECV(P1av:P1bv,P2av,P3bv)=maximum;
imageAD_RECV(P1av:P1bv,P2bv,P3bv)=maximum;
imageAD_RECV(P1av:P1bv,P2bv,P3av)=maximum;
imageAD_RECV(P1av:P1bv,P2av,P3av)=maximum;

imageAD_RECV(P1av,P2av:P2bv,P3bv)=maximum;
imageAD_RECV(P1bv,P2av:P2bv,P3bv)=maximum;
imageAD_RECV(P1av,P2av:P2bv,P3av)=maximum;
imageAD_RECV(P1bv,P2av:P2bv,P3av)=maximum;

imageAD_RECV(P1av,P2av,P3av:P3bv)=maximum;
imageAD_RECV(P1bv,P2av,P3av:P3bv)=maximum;
imageAD_RECV(P1av,P2bv,P3av:P3bv)=maximum;
imageAD_RECV(P1bv,P2bv,P3av:P3bv)=maximum;

figure
subplot(2,2,1)
imagesc(squeeze(imageAD_RECV(:,:,P3av))); colormap gray %la z è fissata, trasversale
title(strcat('Void region - TRASVERSE view - slice= ', num2str(P3av)));
subplot(2,2,3)
imagesc(squeeze(imageAD_RECV(:,P2av,:))); colormap gray %la y è fissata, coronale
title(strcat('Void region - CORONAL view - slice= ', num2str(P2av)));
subplot(2,2,4)
imagesc(squeeze(imageAD_RECV(P1av,:,:))); colormap gray %la x è fissata, sagittale
title(strcat('Void region - SAGITTAL view - slice= ', num2str(P1av)));
