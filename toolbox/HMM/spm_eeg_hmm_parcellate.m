function [Dout,info] = spm_eeg_hmm_parcellate(S)
% Parcel time courses from a DAiSS LCMV beamformer and an atlas
% FORMAT [Dout,info] = spm_eeg_hmm_parcellate(S)
%
% S              - input structure:
%   .BF            DAiSS BF.mat file (or BF structure) with a scalar LCMV
%                  beamformer on a grid (bf_sources grid, bf_inverse lcmv
%                  with orientation to maximum power)
%   .atlas         atlas NIfTI in MNI space (.nii or .nii.gz): 3D labels or
%                  4D non-negative weights (one volume per parcel)
%   .D             M/EEG dataset to parcellate              [BF.data.D]
%   .modality      beamformer modality, e.g. 'MEG'      [first in BF.inverse]
%   .method        'pca' or 'centroid' (see spm_hmm_parcel_weights)  ['pca']
%   .weightnorm    'unit-noise-gain': scale each source filter to unit norm
%                  (unit-noise-gain beamformer for white sensor noise, as
%                  osl-dynamics parcellates weight-normalised filters);
%                  'none': use the DAiSS (unit-gain) filters ['unit-noise-gain']
%   .orthogonalise symmetric orthogonalisation of the parcels afterwards
%                  (spm_eeg_hmm_orthogonalise)                       [false]
%   .prefix        prefix of the output dataset                   ['parc_']
%
% Dout           - float32 dataset with one channel per parcel
%                  ('parcel001', ..., type 'LFP'); events, trials and bad
%                  trials are copied
% info           - .filters  [P x n_channels] parcel filters (sensor space)
%                  .channels sensor channel labels of the filters
%                  .weights  [n_sources x P] (spm_hmm_parcel_weights)
%                  .mni      [n_sources x 3] source positions (mm)
%                  .parcel_weights [n_sources x P] atlas weights
%
% As osl_dynamics.meeg.parcellation.parcellate_lcmv: the covariance of the sources
% is F*C*F', where F are the source filters in sensor space and C the
% sensor covariance (normalised by the number of samples) over the good
% samples of the good trials; parcel filters are V'*F and parcel time
% courses V'*F*(y - mean(y)), with the mean over the same good samples.
% The atlas is sampled at the source positions by nearest neighbour.
% Unlike osl-dynamics, which zero-fills bad samples, the filters are
% applied to every sample and artefact events are kept; an artefact on any
% beamformer sensor applies to all parcels (its event value becomes 'all').
%
% Atlases are not distributed with SPM. Parcellations for MEG (e.g.
% Glasser52, atlas-Glasser_nparc-52_space-MNI_res-8x8x8.nii.gz, CC BY 4.0)
% are in the osl-files repository:
%   https://github.com/OHBA-analysis/osl-files (folder parcellation/;
%   labels and coordinates in docs/parcellations)
% They can also be downloaded with osl-dynamics-download-data or
% osl-ephys-download-data. Cite the authors of the atlas you use.
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


def = struct('D',[],'modality','','method','pca','weightnorm','unit-noise-gain',...
    'orthogonalise',false,'prefix','parc_');
fn = fieldnames(def);
for i = 1:numel(fn)
    if ~isfield(S,fn{i}), S.(fn{i}) = def.(fn{i}); end
end
if ~isfield(S,'BF') || ~isfield(S,'atlas')
    error('spm_hmm:invalidInput','S.BF and S.atlas must be specified.');
end
BF = S.BF;
if ischar(BF), BF = load(BF); end
if isempty(S.D), S.D = BF.data.D; end
D = S.D;
if ~isa(D,'meeg'), D = spm_eeg_load(D); end
mods = fieldnames(BF.inverse);
if isempty(S.modality), S.modality = mods{1}; end
binv = BF.inverse.(S.modality);
U   = BF.features.(S.modality).U;

%-Source filters in sensor space (scalar beamformer only)
%--------------------------------------------------------------------------
ok = cellfun(@(w) ~any(isnan(w(:))),binv.W);
if any(cellfun(@(w) size(w,1),binv.W(ok)) ~= 1)
    error('spm_hmm:invalidInput','A scalar beamformer is needed (orient to maximum power).');
end
F   = cat(1,binv.W{ok})*U';                         % [n_sources x n_channels]
mni = spm_eeg_inv_transform_points(BF.data.transforms.toMNI,BF.sources.pos(ok,:));
switch lower(S.weightnorm)
    case 'unit-noise-gain', F = F./sqrt(sum(F.^2,2));
    case 'none'
    otherwise, error('spm_hmm:invalidInput','Unknown S.weightnorm ''%s''.',S.weightnorm);
end
[ok2,chanind] = ismember(binv.channels,D.chanlabels);
if ~all(ok2), error('spm_hmm:invalidInput','The dataset lacks beamformer channels.'); end
chanind = chanind(:)';

%-Sensor mean and covariance over good samples of good trials
%--------------------------------------------------------------------------
trials = setdiff(1:D.ntrials,D.badtrials);
nc = numel(chanind);
n = 0; s1 = zeros(nc,1); s2 = zeros(nc);
for t = trials
    good = ~meeg_bad_samples(D,chanind,t);
    for lo = 1:1e5:D.nsamples
        hi = min(lo+1e5-1,D.nsamples);
        y  = double(D(chanind,lo:hi,t));
        y  = y(:,good(lo:hi));
        n  = n + size(y,2); s1 = s1 + sum(y,2); s2 = s2 + y*y';
    end
end
if n < 2, error('spm_hmm:invalidInput','Too few good samples.'); end
mu = s1/n;
C  = s2/n - mu*mu';                                 % biased, as np.cov(bias=True)

%-Parcel weights and filters
%--------------------------------------------------------------------------
pw = atlas_weights(S.atlas,mni);
centres = [];
if strcmpi(S.method,'centroid'), centres = spm_hmm_parcel_maps([],S.atlas); end
Vw = spm_hmm_parcel_weights(@(ind) F(ind,:)*C*F(ind,:)',pw,S.method,mni,centres);
Fp = Vw'*F;                                         % [P x n_channels]
P  = size(Fp,1);

%-Write parcel dataset
%--------------------------------------------------------------------------
labels = arrayfun(@(p) sprintf('parcel%03d',p),1:P,'UniformOutput',false);
Dout = output_dataset(D,S.prefix,P);
Dout = parcel_artefact_events(Dout,D,chanind);
Dout = chanlabels(Dout,1:P,labels);
Dout = chantype(Dout,1:P,'LFP');
Dout = units(Dout,1:P,'a.u.');
off  = Fp*mu;
for t = 1:D.ntrials
    for lo = 1:1e5:D.nsamples
        hi = min(lo+1e5-1,D.nsamples);
        Dout(:,lo:hi,t) = Fp*double(D(chanind,lo:hi,t)) - off;
    end
end
Dout = Dout.history(mfilename,S);
save(Dout);

info.filters        = Fp;
info.channels       = binv.channels(:)';
info.weights        = Vw;
info.mni            = mni;
info.parcel_weights = pw;

if S.orthogonalise
    Dout = spm_eeg_hmm_orthogonalise(struct('D',Dout));
end


%==========================================================================
function Dout = parcel_artefact_events(Dout,D,chanind)
% Every parcel combines all beamformer sensors, so an artefact on any of
% them marks the sample bad for all parcels: such events get value 'all'
for t = 1:D.ntrials
    ev = D.events(t);
    if iscell(ev), ev = ev{1}; end
    if isempty(ev), continue; end
    for k = 1:numel(ev)
        if strncmp(ev(k).type,'artefact',8) && ischar(ev(k).value) && ...
                ~strcmpi(ev(k).value,'all') && ...
                any(ismember(D.selectchannels(ev(k).value),chanind))
            ev(k).value = 'all';
        end
    end
    Dout = events(Dout,t,ev);
end
