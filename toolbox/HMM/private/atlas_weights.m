function [w,nparc] = atlas_weights(file,mni)
% Parcel weights of points sampled from an atlas (nearest neighbour)
% FORMAT [w,nparc] = atlas_weights(file,mni)
%
% file  - atlas NIfTI (.nii or .nii.gz): 3D labels (one parcel per unique
%         non-zero value, in ascending order) or 4D weights (one volume per
%         parcel, non-negative)
% mni   - [n x 3] coordinates in mm
%
% w     - [n x nparc] weights; points outside the atlas get zero
%
% As osl-ephys parcellation._sample_parcellation: no resampling, each point
% takes the value of its nearest atlas voxel.
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


[V,cleanup] = open_atlas(file); %#ok<ASGLU>
Y = spm_read_vols(V);
if any(Y(:) < 0)
    error('spm_hmm:invalidInput','The weights in %s must be non-negative.',file);
end
if isscalar(V)                                   % labels -> one-hot
    lab = unique(Y(Y ~= 0));
    nparc = numel(lab);
else
    nparc = numel(V);
end

vox = [mni ones(size(mni,1),1)]/V(1).mat';         % 1-based voxel coordinates
ijk = round(vox(:,1:3));
dim = V(1).dim(1:3);
in  = all(ijk >= 1 & ijk <= dim,2);
w   = zeros(size(mni,1),nparc);
ind = sub2ind(dim,ijk(in,1),ijk(in,2),ijk(in,3));
if isscalar(V)
    y = Y(ind);
    [~,col] = ismember(y,lab);
    rows = find(in);
    k = col > 0;
    w(sub2ind(size(w),rows(k),col(k))) = 1;
else
    Y = reshape(Y,[],nparc);
    w(in,:) = Y(ind,:);
end


%==========================================================================
function [V,cleanup] = open_atlas(file)
cleanup = [];
if numel(file) > 3 && strcmpi(file(end-2:end),'.gz')
    d = tempname; mkdir(d);
    cleanup = onCleanup(@() rmdir(d,'s'));
    file = char(gunzip(file,d));
end
V = spm_vol(file);
