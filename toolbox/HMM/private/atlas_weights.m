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
% each point takes the value of its nearest atlas voxel (ties at exactly
% half a voxel rounded to even, as numpy.rint).
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


A     = read_atlas(file);
nparc = A.P;
% 0-based voxel coordinates (as nibabel's affine), rounded half to even
% (numpy.rint), then 1-based indices
A0    = A.mat*[eye(3) ones(3,1); 0 0 0 1];        % 0-based voxel-to-mm
iA    = inv(A0);
v     = mni*iA(1:3,1:3)' + iA(1:3,4)';
r     = round(v);
tie   = (v - floor(v)) == 0.5;
r(tie) = 2*round(v(tie)/2);
ijk   = r + 1;
in    = all(ijk >= 1 & ijk <= A.dim,2);
w     = zeros(size(mni,1),nparc);
ind   = sub2ind(A.dim,ijk(in,1),ijk(in,2),ijk(in,3));
w(in,:) = full(A.W(ind,:));
