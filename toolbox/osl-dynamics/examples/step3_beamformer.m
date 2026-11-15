%% Step 3: LCMV beamformer (DAiSS)
%
% Sources on an 8 mm grid of the MNI template inside the inner skull; the
% data covariance over the good samples of the run (DAiSS 'contcov', for
% continuous data: the mean of each 1 s piece is removed), truncated to its
% 100 largest components; scalar LCMV filters with each source oriented to
% maximum power. The filters are
% normalised (unit noise gain) when the parcels are formed in step4_parcellate.
%
% Output (in cfg.outdir/sessions): <session>/BF/BF.mat
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


if ~exist('cfg','var'), cfg = osld_example_config(); end

for s = 1:numel(cfg.sessions)
    id  = cfg.sessions(s).id;
    bfd = fullfile(cfg.sessiondir,id,'BF');
    if ~exist(bfd,'dir'), mkdir(bfd); end
    matlabbatch = {};
    matlabbatch{1}.spm.tools.beamforming.data.dir = {bfd};
    matlabbatch{1}.spm.tools.beamforming.data.D = {fullfile(cfg.sessiondir,id,[id '.mat'])};
    matlabbatch{1}.spm.tools.beamforming.data.val = 1;
    matlabbatch{1}.spm.tools.beamforming.data.gradsource = 'inv';
    matlabbatch{1}.spm.tools.beamforming.data.space = 'MNI-aligned';
    matlabbatch{1}.spm.tools.beamforming.data.overwrite = 1;
    matlabbatch{2}.spm.tools.beamforming.sources.BF = {fullfile(bfd,'BF.mat')};
    matlabbatch{2}.spm.tools.beamforming.sources.reduce_rank = [2 3];
    matlabbatch{2}.spm.tools.beamforming.sources.keep3d = 1;
    matlabbatch{2}.spm.tools.beamforming.sources.plugin.grid.resolution = 8;
    matlabbatch{2}.spm.tools.beamforming.sources.plugin.grid.space = 'MNI template';
    matlabbatch{2}.spm.tools.beamforming.sources.plugin.grid.constrain = 'iskull';
    matlabbatch{2}.spm.tools.beamforming.sources.visualise = 0;
    matlabbatch{3}.spm.tools.beamforming.features.BF = {fullfile(bfd,'BF.mat')};
    matlabbatch{3}.spm.tools.beamforming.features.whatconditions.all = 1;
    matlabbatch{3}.spm.tools.beamforming.features.woi = [-Inf Inf];
    matlabbatch{3}.spm.tools.beamforming.features.modality = {'MEG'};
    matlabbatch{3}.spm.tools.beamforming.features.fuse = 'no';
    matlabbatch{3}.spm.tools.beamforming.features.plugin.contcov = struct();
    matlabbatch{3}.spm.tools.beamforming.features.regularisation.mantrunc.pcadim = 100;
    matlabbatch{3}.spm.tools.beamforming.features.bootstrap = false;
    matlabbatch{4}.spm.tools.beamforming.inverse.BF = {fullfile(bfd,'BF.mat')};
    matlabbatch{4}.spm.tools.beamforming.inverse.plugin.lcmv.orient = true;
    matlabbatch{4}.spm.tools.beamforming.inverse.plugin.lcmv.keeplf = false;
    spm_jobman('run',matlabbatch);
    BF = load(fullfile(bfd,'BF.mat'));
    fprintf('%s: %d sources, filters for %d channels\n',id,numel(BF.inverse.MEG.W),size(BF.inverse.MEG.W{1},2));
end
