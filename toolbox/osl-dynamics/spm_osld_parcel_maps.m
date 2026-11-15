function [V,centres] = spm_osld_parcel_maps(values,atlas,fname)
% Write parcel values as a NIfTI map on the atlas grid; parcel centres
% FORMAT [V,centres] = spm_osld_parcel_maps(values,atlas,fname)
% FORMAT centres = spm_osld_parcel_maps([],atlas)
%
% values   - [P x N] values of each parcel for N maps (e.g. power maps of
%            N states), or [] to return only the parcel centres
% atlas    - atlas NIfTI (.nii or .nii.gz; 3D labels or 4D weights)
% fname    - output NIfTI file (.nii or .nii.gz; 3D for N = 1, 4D otherwise)
%
% V        - header(s) of the written map
% centres  - [P x 3] weighted centre of each parcel (mm)
%
% As osl_dynamics.meeg.parcellation.parcel_vector_to_nifti: each parcel's
% voxel weights are divided by their maximum, and the value of a voxel is
% the weighted sum of the parcel values. Parcel centres are the weighted
% mean of the coordinates of each parcel's non-zero voxels
% (Parcellation.roi_centers).
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


A = read_atlas(atlas);
nz = find(any(A.W,2));                                % voxels in any parcel
[i,j,k] = ind2sub(A.dim,nz);
xyz = [i j k ones(numel(i),1)]*A.mat';
xyz = xyz(:,1:3);
Wn  = A.W(nz,:);
centres = full((Wn'*xyz)./sum(Wn,1)');
if isempty(values)
    V = centres;
    return
end
if size(values,1) ~= A.P
    error('spm_osld:invalidInput','The atlas has %d parcels, values has %d rows.',A.P,size(values,1));
end
mx = full(max(A.W,[],1)); mx(mx == 0) = 1;
Y  = full((A.W*spdiags(1./mx',0,A.P,A.P))*values);   % [nvox x N]
N  = size(values,2);
gz = numel(fname) > 3 && strcmpi(fname(end-2:end),'.gz');
if gz, fname = fname(1:end-3); end
V0 = struct('fname',fname,'dim',A.dim,'mat',A.mat,'dt',[spm_type('float32') spm_platform('bigend')],...
    'pinfo',[1;0;0],'descrip','spm_osld_parcel_maps');
V  = cell(N,1);
for n = 1:N
    V0.n = [n 1];
    V{n} = spm_write_vol(V0,reshape(Y(:,n),A.dim));
end
V = cat(1,V{:});
if gz                                                  % write .nii, then gzip
    gzip(fname); delete(fname);
    [V.fname] = deal([fname '.gz']);
end
