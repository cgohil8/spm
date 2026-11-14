%% Parcel-level HMM, step 2: coregistration and forward model
%
% Each session is coregistered to SPM's template head by matching its three
% head coils to the template's fiducials (cfg.fiducials: the nasion and the
% preauricular points), and a single-shell forward model is computed
% (Nolte 2003; inner skull of the template). With a structural MRI or a
% head shape, specify them in the coregistration instead.
%
% Output: the head model is stored in each dataset (D.inv).
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


addpath(fullfile(fileparts(mfilename('fullpath')),'..'));
if ~exist('cfg','var'), cfg = hmm_example_config(); end

for s = 1:numel(cfg.sessions)
    id = cfg.sessions(s).id;
    matlabbatch = {};
    h.D = {fullfile(cfg.parceldir,id,[id '.mat'])};
    h.val = 1;
    h.comment = '';
    h.meshing.meshes.template = 1;
    h.meshing.meshres = 2;
    h.coregistration.coregspecify.fiducial(1).fidname = 'nas';
    h.coregistration.coregspecify.fiducial(1).specification.select = cfg.fiducials{1};
    h.coregistration.coregspecify.fiducial(2).fidname = 'lpa';
    h.coregistration.coregspecify.fiducial(2).specification.select = cfg.fiducials{2};
    h.coregistration.coregspecify.fiducial(3).fidname = 'rpa';
    h.coregistration.coregspecify.fiducial(3).specification.select = cfg.fiducials{3};
    h.coregistration.coregspecify.useheadshape = 0;
    h.forward.eeg = 'EEG BEM';
    h.forward.meg = 'Single Shell';
    matlabbatch{1}.spm.meeg.source.headmodel = h;
    spm_jobman('run',matlabbatch);

    D = spm_eeg_load(h.D{1});
    reg = D.inv{1}.datareg(1);                  % both sets in head space
    err = sqrt(sum((reg.fid_eeg.fid.pnt - reg.fid_mri.fid.pnt).^2,2));
    fprintf('%s: coil-template distances after coregistration (nas, lpa, rpa): %s mm\n',...
        id,mat2str(err',2));
end
