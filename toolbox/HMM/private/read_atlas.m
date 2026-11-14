function A = read_atlas(file)
% Read an atlas as 4D parcel weights
% FORMAT A = read_atlas(file)
%
% file  - NIfTI (.nii or .nii.gz): 3D labels (one parcel per unique
%         non-zero value, in ascending order) or 4D non-negative weights
%
% A     - .W   [nvox x P] weights (columns: parcels), voxels in the order of
%              the 3D grid
%         .dim [1 x 3] grid size, .mat voxel-to-mm matrix (1-based voxels)
%         .P   number of parcels
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


if numel(file) > 3 && strcmpi(file(end-2:end),'.gz')
    d = tempname; mkdir(d);
    cleanup = onCleanup(@() rmdir(d,'s'));
    file = char(gunzip(file,d));
end
V = spm_vol(file);
Y = spm_read_vols(V);
if any(Y(:) < 0)
    error('spm_hmm:invalidInput','The weights in %s must be non-negative.',file);
end
dim = V(1).dim(1:3);
if isscalar(V)
    lab = unique(Y(Y ~= 0));
    A.W = double(Y(:) == lab');
else
    A.W = reshape(Y,prod(dim),numel(V));
end
A.dim = dim;
A.mat = V(1).mat;
A.P   = size(A.W,2);
