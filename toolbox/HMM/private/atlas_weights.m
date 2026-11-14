function [w,nparc] = atlas_weights(file,mni)
% Parcel weights of points sampled from an atlas (nearest neighbour)
% FORMAT [w,nparc] = atlas_weights(file,mni)
%
% file  - atlas NIfTI (.nii or .nii.gz): 3D labels or 4D weights (see
%         read_atlas)
% mni   - [n x 3] coordinates in mm
%
% w     - [n x nparc] weights; points outside the atlas get zero
%
% As osl_dynamics.meeg.parcellation._sample_parcellation: no resampling,
% each point takes the value of its nearest atlas voxel.
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


A     = read_atlas(file);
nparc = A.P;
vox   = [mni ones(size(mni,1),1)]/A.mat';          % 1-based voxel coordinates
ijk   = round(vox(:,1:3));
in    = all(ijk >= 1 & ijk <= A.dim,2);
w     = zeros(size(mni,1),nparc);
ind   = sub2ind(A.dim,ijk(in,1),ijk(in,2),ijk(in,3));
w(in,:) = A.W(ind,:);
