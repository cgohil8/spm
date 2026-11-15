function [HMM,Dout] = spm_osld_hmm(S)
% Fit a hidden Markov model to M/EEG data with osl-dynamics
% FORMAT [HMM,Dout] = spm_osld_hmm(S)
%
% S             - input structure:
%   .D            meeg object or file name, or a cell array of these
%                 (e.g. one per session or subject; one group-level model)
%   .K            number of states                               [required]
%   .channels     channel labels; overrides .chantype                  [{}]
%   .chantype     channel types, good channels only [{'MEEG','LFP','SRC'}]
%   .conditions   condition labels of the trials to use      [all trials]
%   .standardise  z-score the channels of each dataset              [true]
%   .pca          number of principal components of the channels to keep
%                 before any embedding, e.g. to reduce many sensors;
%                 z-scored within each dataset (with .standardise) when
%                 there is no embedding                                [[]]
%   .tde          time-delay embedding and PCA: struct with .lags
%                 (symmetric, e.g. -7:7) and .npca (e.g. 80); components
%                 are z-scored within each dataset (with .standardise)
%                                                                [struct()]
%   .zeromean     zero-mean states            [true with .tde, else false]
%   .hmm          options of osl_dynamics.models.hmm.Config, e.g.
%                 .sequence_length [200], .batch_size [256],
%                 .learning_rate [0.01], .n_epochs [20], .n_init [5],
%                 .n_init_epochs [2], .best_of [1]
%   .seed         random seed                                          [[]]
%   .modeldir     folder to save the trained osl-dynamics model in     ['']
%   .write        write the state probabilities to new datasets      [true]
%   .prefix       prefix of the written datasets                  ['hmm_']
%   .python       Python executable                       [spm_osld_python]
%
% HMM           - fitted model:
%   .K
%   .F            osl-dynamics' variational free energy (lower is better)
%   .Ep           .pi [K x 1] initial state probabilities, .A [K x K]
%                 transition probabilities (row i: from state i), .mu
%                 [D x K] means and .Sigma [D x D x K] covariances of the
%                 states, in the space of the prepared data
%   .path         {datasets x 1} Viterbi paths of the modelled samples
%   .missing      {datasets x 1} true for the samples at the ends of each
%                 dataset lost by time-delay embedding
%   .stats        summary statistics of the Viterbi paths, [datasets x K]:
%                 .fo (fractional occupancy), .mean_lifetime (s),
%                 .mean_interval (s), .switching_rate (per s)
%   .meeg         see meeg_read: .files, .fsample, .chanlabels, .index
%                 (samples of each dataset that were modelled)
%   .pca, .tde    principal components (.W) of each preparation step
%   .osld         .config (osl-dynamics configuration), .versions
% Dout          - cell array of the written datasets (if S.write): channels
%                 'State1'..'StateK' (type 'Other') hold the state
%                 probabilities; samples that were not modelled are NaN
%
% Each dataset is one session: its good samples (of its good trials) are
% concatenated, leaving out bad samples (artefact events), as osl-dynamics
% does for annotated bad segments. The data are prepared (z-scoring, PCA,
% time-delay embedding and PCA) and the HMM is trained with osl-dynamics
% (osl_dynamics.models.hmm: stochastic variational inference of the state
% time courses, with point estimates of the observation model).
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


%-Options
%--------------------------------------------------------------------------
if ~isfield(S,'D') || isempty(S.D), error('spm_osld:invalidInput','S.D is required.'); end
if ~isfield(S,'K') || isempty(S.K), error('spm_osld:invalidInput','S.K is required.'); end
def = struct('channels',{{}},'chantype',{{'MEEG','LFP','SRC'}},'conditions',{{}}, ...
    'standardise',true,'pca',[],'tde',struct(),'zeromean',[],'hmm',struct(), ...
    'seed',[],'modeldir','','write',true,'prefix','hmm_','python','');
fn = fieldnames(def);
for i = 1:numel(fn)
    if ~isfield(S,fn{i}), S.(fn{i}) = def.(fn{i}); end
end
tde = isfield(S.tde,'lags') && ~isempty(S.tde.lags);
if tde
    lags = S.tde.lags(:)';
    if ~isequal(lags,-max(lags):max(lags)) || max(lags) < 1
        error('spm_osld:invalidInput','S.tde.lags must be -L:L (osl-dynamics embeds symmetrically).');
    end
    if ~isfield(S.tde,'npca'), S.tde.npca = []; end
end
if isempty(S.zeromean), S.zeromean = tde; end

%-Data
%--------------------------------------------------------------------------
[X,info] = meeg_read(S.D,S);
fprintf('spm_osld_hmm: %d dataset(s), %d channels, %d samples\n', ...
    numel(X),numel(info.chanlabels),sum(cellfun(@(x) size(x,1),X)));

%-Fit with osl-dynamics
%--------------------------------------------------------------------------
args = struct('data',{X},'sampling_frequency',info.fsample,'n_states',S.K, ...
    'learn_means',~S.zeromean,'config',S.hmm,'standardize',logical(S.standardise), ...
    'n_pca_components',S.pca,'seed',S.seed,'model_dir',S.modeldir);
if tde
    args.n_embeddings = numel(lags);
    args.n_tde_pca_components = S.tde.npca;
end
if isempty(fieldnames(args.config)), args = rmfield(args,'config'); end
r = osld_run('fit_hmm',args,struct('python',S.python));

%-Model
%--------------------------------------------------------------------------
alpha = as_cell(r.alpha);
path  = as_cell(r.path);
HMM.K  = S.K;
HMM.F  = r.free_energy;
HMM.Ep = struct('pi',r.initial_state_probs(:),'A',r.trans_prob,'mu',r.means,'Sigma',r.covs);
if S.zeromean, HMM.Ep.mu = zeros(size(r.means)); end
HMM.path    = cell(numel(X),1);
HMM.missing = cell(numel(X),1);
for f = 1:numel(X)
    HMM.missing{f} = path{f}(:) == 0;
    HMM.path{f}    = max(path{f}(:),1);
end
HMM.stats = r.stats;
HMM.meeg  = info;
if isfield(r,'pca_components'), HMM.pca.W = r.pca_components; end
if tde, HMM.tde = struct('lags',lags,'npca',S.tde.npca,'W',r.tde_pca_components); end
HMM.osld.config   = r.config;
HMM.osld.versions = osld_run('versions',struct(),struct('python',S.python));

%-State probabilities as datasets
%--------------------------------------------------------------------------
Dout = {};
if S.write
    Dout = cell(numel(X),1);
    for f = 1:numel(X)
        D = spm_eeg_load(info.files{f});
        Dk = output_dataset(D,S.prefix,S.K);
        Dk = chanlabels(Dk,1:S.K,arrayfun(@(k) sprintf('State%d',k),1:S.K,'UniformOutput',false));
        Dk = chantype(Dk,1:S.K,'Other');
        Dk = units(Dk,1:S.K,'');
        Dk = badchannels(Dk,1:S.K,0);
        a = NaN(S.K,info.nsamples(f)*info.ntrials(f));
        a(:,info.index{f}) = alpha{f}';
        Dk(:,:,:) = reshape(a,S.K,info.nsamples(f),info.ntrials(f));
        save(Dk);
        Dout{f} = Dk;
    end
end


function x = as_cell(x)
if ~iscell(x), x = {x}; end
