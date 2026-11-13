function hmm = tbx_cfg_hmm
% Batch configuration for the HMM toolbox (SPM > Tools > HMM)
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


if ~isdeployed, addpath(fileparts(mfilename('fullpath'))); end

%--------------------------------------------------------------------------
% Data and channels
%--------------------------------------------------------------------------
D        = cfg_files;
D.tag    = 'D';
D.name   = 'M/EEG datasets';
D.filter = 'mat';
D.num    = [1 Inf];
D.help   = {['Select the M/EEG datasets (e.g. one per session or subject). '...
    'One group-level model is fitted to all of them.']};

default_channels      = cfg_const;
default_channels.tag  = 'default';
default_channels.name = 'Good MEEG, LFP and SRC channels';
default_channels.val  = {true};
default_channels.help = {'All good channels of type MEG, EEG, LFP or SRC.'};

selected_channels      = spm_cfg_eeg_channel_selector;
selected_channels.tag  = 'selected';
selected_channels.name = 'Select channels';

channels        = cfg_choice;
channels.tag    = 'channels';
channels.name   = 'Channels';
channels.values = {default_channels, selected_channels};
channels.val    = {default_channels};
channels.help   = {'Channels to model. Bad channels are excluded.'};

%--------------------------------------------------------------------------
% Model
%--------------------------------------------------------------------------
K         = cfg_entry;
K.tag     = 'K';
K.name    = 'Number of states';
K.strtype = 'n';
K.num     = [1 1];
K.val     = {8};
K.help    = {'Number of hidden states.'};

lifetime         = cfg_entry;
lifetime.tag     = 'lifetime';
lifetime.name    = 'Expected state lifetime (s)';
lifetime.strtype = 'r';
lifetime.num     = [1 1];
lifetime.val     = {0.1};
lifetime.help    = {['Prior expected duration of a state visit, which sets '...
    'the prior on the self-transition probabilities.']};

zeromean        = cfg_menu;
zeromean.tag    = 'zeromean';
zeromean.name   = 'State means';
zeromean.labels = {'Default (zero with embedding, learned otherwise)','Learned','Zero'};
zeromean.values = {-1, 0, 1};
zeromean.val    = {-1};
zeromean.help   = {'Whether states have learned means or are zero-mean.'};

no_tde      = cfg_const;
no_tde.tag  = 'none';
no_tde.name = 'None';
no_tde.val  = {true};

lags         = cfg_entry;
lags.tag     = 'lags';
lags.name    = 'Lags (samples)';
lags.strtype = 'i';
lags.num     = [1 Inf];
lags.val     = {-7:7};
lags.help    = {'Integer lags of the time-delay embedding.'};

npca         = cfg_entry;
npca.tag     = 'npca';
npca.name    = 'PCA components';
npca.strtype = 'r';
npca.num     = [1 1];
npca.val     = {80};
npca.help    = {['Number of principal components, or the fraction of '...
    'variance to keep if between 0 and 1.']};

tde      = cfg_branch;
tde.tag  = 'tde';
tde.name = 'Time-delay embedding + PCA';
tde.val  = {lags, npca};

embedding        = cfg_choice;
embedding.tag    = 'embedding';
embedding.name   = 'Embedding';
embedding.values = {no_tde, tde};
embedding.val    = {no_tde};
embedding.help   = {'Optional time-delay embedding and PCA (TDE-HMM).'};

%--------------------------------------------------------------------------
% Inference
%--------------------------------------------------------------------------
batch      = cfg_const;
batch.tag  = 'batch';
batch.name = 'Batch variational Bayes';
batch.val  = {true};

nepochs         = cfg_entry;
nepochs.tag     = 'nepochs';
nepochs.name    = 'Epochs';
nepochs.strtype = 'n';
nepochs.num     = [1 1];
nepochs.val     = {20};

chunklength         = cfg_entry;
chunklength.tag     = 'chunklength';
chunklength.name    = 'Chunk length (samples)';
chunklength.strtype = 'n';
chunklength.num     = [1 1];
chunklength.val     = {1000};

batchsize         = cfg_entry;
batchsize.tag     = 'batchsize';
batchsize.name    = 'Chunks per mini-batch';
batchsize.strtype = 'n';
batchsize.num     = [1 1];
batchsize.val     = {32};

stochastic      = cfg_branch;
stochastic.tag  = 'stochastic';
stochastic.name = 'Stochastic variational inference';
stochastic.val  = {nepochs, chunklength, batchsize};

inference        = cfg_choice;
inference.tag    = 'inference';
inference.name   = 'Inference';
inference.values = {batch, stochastic};
inference.val    = {batch};
inference.help   = {'Batch VB, or stochastic VB for large datasets.'};

bestof         = cfg_entry;
bestof.tag     = 'bestof';
bestof.name    = 'Complete fits';
bestof.strtype = 'n';
bestof.num     = [1 1];
bestof.val     = {1};
bestof.help    = {'Number of complete fits; the one with the highest free energy is kept.'};

seed         = cfg_entry;
seed.tag     = 'seed';
seed.name    = 'Random seed';
seed.strtype = 'n';
seed.num     = [1 1];
seed.val     = {1};

%--------------------------------------------------------------------------
% Output
%--------------------------------------------------------------------------
prefix         = cfg_entry;
prefix.tag     = 'prefix';
prefix.name    = 'Output prefix';
prefix.strtype = 's';
prefix.num     = [1 Inf];
prefix.val     = {'hmm_'};
prefix.help    = {['Prefix of the datasets holding the state probabilities, '...
    'and of the model file (<prefix>model.mat, next to the first dataset).']};

%--------------------------------------------------------------------------
% Executable branch
%--------------------------------------------------------------------------
hmm          = cfg_exbranch;
hmm.tag      = 'hmm';
hmm.name     = 'HMM';
hmm.val      = {D, channels, K, lifetime, zeromean, embedding, inference, bestof, seed, prefix};
hmm.help     = {['Variational Bayesian hidden Markov model with Gaussian '...
    'observations (see toolbox/HMM/doc/HMM_maths.md).']};
hmm.prog     = @run_hmm;
hmm.vout     = @vout_hmm;
hmm.modality = {'EEG'};


%==========================================================================
function out = run_hmm(job)
S          = struct();
S.D        = cellstr(job.D);
S.K        = job.K;
S.lifetime = job.lifetime;
S.prefix   = job.prefix;
if isfield(job.channels,'selected')
    D1 = spm_eeg_load(S.D{1});
    ind = D1.selectchannels(spm_cfg_eeg_channel_selector(job.channels.selected));
    S.channels = D1.chanlabels(setdiff(ind,D1.badchannels,'stable'));
end
if isfield(job.embedding,'tde')
    S.tde = struct('lags',job.embedding.tde.lags,'npca',job.embedding.tde.npca);
end
S.hmm = struct('bestof',job.bestof,'seed',job.seed);
if job.zeromean >= 0, S.hmm.zeromean = logical(job.zeromean); end
if isfield(job.inference,'stochastic')
    st = job.inference.stochastic;
    S.hmm.inference = 'stochastic';
    S.hmm.svi = struct('nepochs',st.nepochs,'chunklength',st.chunklength,'batchsize',st.batchsize);
end

[HMM,Dout] = spm_eeg_hmm(S);

modelfile = spm_file(S.D{1},'filename',[job.prefix 'model.mat']);
save(modelfile,'HMM',spm_get_defaults('mat.format'));
out.D      = Dout;
out.Dfname = cellfun(@fullfile,Dout,'UniformOutput',false);
out.model  = {modelfile};


%==========================================================================
function dep = vout_hmm(~)
dep(1)            = cfg_dep;
dep(1).sname      = 'State probability datasets';
dep(1).src_output = substruct('.','Dfname');
dep(1).tgt_spec   = cfg_findspec({{'filter','mat'}});
dep(2)            = cfg_dep;
dep(2).sname      = 'HMM model file';
dep(2).src_output = substruct('.','model');
dep(2).tgt_spec   = cfg_findspec({{'filter','mat'}});
