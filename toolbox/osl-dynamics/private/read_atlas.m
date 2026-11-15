function A = read_atlas(file)
% Read an atlas as parcel weights (sparse)
% FORMAT A = read_atlas(file)
%
% file  - NIfTI (.nii or .nii.gz): a 3D image of labels (one parcel per
%         unique non-zero value, in ascending order) or a 4D image of
%         non-negative weights (one volume per parcel, also when it has a
%         single volume). NaN is treated as 0.
%
% A     - .W   [nvox x P] sparse weights (columns: parcels), voxels in the
%              order of the 3D grid
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
V   = spm_vol(file);
dim = V(1).dim(1:3);
nv  = prod(dim);
is4d = numel(V) > 1 || numel(V(1).private.dat.dim) > 3;
if ~is4d
    Y = spm_read_vols(V);
    Y(isnan(Y)) = 0;
    check_nonnegative(Y,file);
    ind = find(Y);
    [lab,~,col] = unique(Y(ind));
    A.W = sparse(ind,col,1,nv,numel(lab));
else
    W = cell(1,numel(V));
    for p = 1:numel(V)                         % one volume at a time
        y = spm_read_vols(V(p));
        y(isnan(y)) = 0;
        check_nonnegative(y,file);
        W{p} = sparse(y(:));
    end
    A.W = [W{:}];
end
A.dim = dim;
A.mat = V(1).mat;
A.P   = size(A.W,2);


%==========================================================================
function check_nonnegative(Y,file)
if any(Y(:) < 0)
    error('spm_osld:invalidInput','The weights in %s must be non-negative.',file);
end
