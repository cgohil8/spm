function osld = tbx_cfg_osld
% Batch configuration of the OSLDynamics toolbox
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


if ~isdeployed, addpath(fileparts(mfilename('fullpath'))); end

osld        = cfg_choice;
osld.tag    = 'osld';
osld.name   = 'OSL-Dynamics';
osld.help   = {['Dynamic network models of M/EEG data with osl-dynamics. ' ...
    'Install osl-dynamics first with spm_osld_install.']};
osld.values = {orthogonalise_cfg, sign_flip_cfg, hmm_cfg};


%==========================================================================
% Modules
%==========================================================================
function m = orthogonalise_cfg
m          = cfg_exbranch;
m.tag      = 'orthogonalise';
m.name     = 'Leakage correction';
m.val      = {meeg_files('D',[1 Inf]), chantype_entry, prefix_entry('orth_')};
m.help     = {['Symmetric orthogonalisation of the channels of each dataset ' ...
    '(e.g. parcel time courses), which removes the zero-lag correlations caused by ' ...
    'spatial leakage (Colclough et al. 2015). Each channel keeps its norm. The data ' ...
    'should be demeaned (as DAiSS parcel time courses are).']};
m.prog     = @run_orthogonalise;
m.vout     = @(job) meeg_dep('Orthogonalised datasets');
m.modality = {'EEG'};


function m = sign_flip_cfg
nembed         = cfg_entry;
nembed.tag     = 'nembed';
nembed.name    = 'Number of embeddings';
nembed.help    = {['Number of time-delay embeddings (odd) of the covariance used ' ...
    'to compare the sessions, e.g. as many as the lags of the HMM.']};
nembed.strtype = 'n';
nembed.num     = [1 1];
nembed.val     = {15};

template        = cfg_files;
template.tag    = 'template';
template.name   = 'Template covariance';
template.help   = {['Optional .mat file with a variable ''template'': a template ' ...
    'covariance of the time-delay embedded data. Without it, the sessions are ' ...
    'aligned to the median session.']};
template.filter = 'mat';
template.num    = [0 1];
template.val    = {{''}};

twostage        = cfg_menu;
twostage.tag    = 'twostage';
twostage.name   = 'Two stages';
twostage.help   = {['With a template: first align the sessions to a template ' ...
    'built from them, then align that group to the template.']};
twostage.labels = {'Yes','No'};
twostage.values = {true,false};
twostage.val    = {false};

nrandom         = cfg_entry;
nrandom.tag     = 'nrandom';
nrandom.name    = 'Random starts';
nrandom.help    = {'Number of random starts of the search for the signs of each session.'};
nrandom.strtype = 'n';
nrandom.num     = [1 1];
nrandom.val     = {3};

m          = cfg_exbranch;
m.tag      = 'sign_flip';
m.name     = 'Sign flipping';
m.val      = {meeg_files('D',[1 Inf]), chantype_entry, nembed, template, twostage, ...
    nrandom, prefix_entry('sf_')};
m.help     = {['Aligns the signs of the channels (e.g. parcels) across sessions, ' ...
    'whose signs are arbitrary after source reconstruction, using their time-delay ' ...
    'embedded covariance. Needed before a time-delay embedded HMM of several sessions.']};
m.prog     = @run_sign_flip;
m.vout     = @(job) meeg_dep('Sign-flipped datasets');
m.modality = {'EEG'};


function m = hmm_cfg
K         = cfg_entry;
K.tag     = 'K';
K.name    = 'Number of states';
K.strtype = 'n';
K.num     = [1 1];
K.val     = {8};
K.help    = {'Number of HMM states.'};

standardise        = cfg_menu;
standardise.tag    = 'standardise';
standardise.name   = 'Standardise';
standardise.help   = {['z-score the channels of each dataset (and the principal ' ...
    'components after PCA or time-delay embedding).']};
standardise.labels = {'Yes','No'};
standardise.values = {true,false};
standardise.val    = {true};

pca         = cfg_entry;
pca.tag     = 'pca';
pca.name    = 'Principal components of the channels';
pca.help    = {['Number of principal components of the channels to keep before ' ...
    'any embedding, e.g. to reduce many sensors; 0 for no PCA.']};
pca.strtype = 'w';
pca.num     = [1 1];
pca.val     = {0};

none      = cfg_const;
none.tag  = 'none';
none.name = 'None';
none.val  = {true};
none.help = {'Model the (prepared) channels directly.'};

lags         = cfg_entry;
lags.tag     = 'lags';
lags.name    = 'Lags';
lags.help    = {'Number of lags L on each side: the data are embedded with lags -L:L (samples).'};
lags.strtype = 'n';
lags.num     = [1 1];
lags.val     = {7};

npca         = cfg_entry;
npca.tag     = 'npca';
npca.name    = 'Principal components';
npca.help    = {'Number of principal components of the embedded data to keep.'};
npca.strtype = 'n';
npca.num     = [1 1];
npca.val     = {80};

tde      = cfg_branch;
tde.tag  = 'tde';
tde.name = 'Time-delay embedding';
tde.val  = {lags, npca};
tde.help = {['Time-delay embedding followed by PCA. The states have zero mean and ' ...
    'differ in the covariance of the embedded data, i.e. in their power spectra and ' ...
    'coherence.']};

embedding        = cfg_choice;
embedding.tag    = 'embedding';
embedding.name   = 'Embedding';
embedding.values = {none, tde};
embedding.val    = {tde};
embedding.help   = {'Time-delay embedding of the data.'};

means        = cfg_menu;
means.tag    = 'means';
means.name   = 'State means';
means.help   = {'Whether the states have their own means (zero-mean states otherwise).'};
means.labels = {'Default (zero with embedding, learnt otherwise)','Learn','Zero'};
means.values = {'default','learn','zero'};
means.val    = {'default'};

training      = cfg_branch;
training.tag  = 'training';
training.name = 'Training';
training.val  = {int_entry('n_epochs','Epochs',20,'Number of training epochs.'), ...
    int_entry('n_init','Initialisations',5, ...
        'Number of random initialisations; the best is trained further.'), ...
    int_entry('n_init_epochs','Initialisation epochs',2, ...
        'Number of epochs of each initialisation.'), ...
    int_entry('sequence_length','Sequence length',200, ...
        'Length (samples) of the sequences the data are split into.'), ...
    int_entry('batch_size','Batch size',256,'Number of sequences per batch.'), ...
    real_entry('learning_rate','Learning rate',0.01,'Learning rate of the optimiser.')};
training.help = {'Training options of osl_dynamics.models.hmm.'};

seed         = cfg_entry;
seed.tag     = 'seed';
seed.name    = 'Random seed';
seed.help    = {'Random seed, for reproducible results; empty for none.'};
seed.strtype = 'w';
seed.num     = [0 1];
seed.val     = {[]};

outdir        = cfg_files;
outdir.tag    = 'outdir';
outdir.name   = 'Output directory';
outdir.help   = {'Folder where the model is saved.'};
outdir.filter = 'dir';
outdir.num    = [1 1];

name         = cfg_entry;
name.tag     = 'name';
name.name    = 'Model name';
name.help    = {'The model is saved as <name>.mat (variable HMM) in the output directory.'};
name.strtype = 's';
name.num     = [1 Inf];
name.val     = {'hmm'};

m          = cfg_exbranch;
m.tag      = 'hmm';
m.name     = 'Hidden Markov model';
m.val      = {meeg_files('D',[1 Inf]), chantype_entry, K, standardise, pca, embedding, ...
    means, training, seed, outdir, name, prefix_entry('hmm_')};
m.help     = {['Fits one group-level hidden Markov model to all datasets (e.g. one ' ...
    'per session) with osl-dynamics. The good samples of each dataset are modelled; ' ...
    'the state probabilities are written as new datasets (NaN where not modelled) and ' ...
    'the model, with its Viterbi paths and summary statistics, is saved as a .mat file.']};
m.prog     = @run_hmm;
m.vout     = @vout_hmm;
m.modality = {'EEG'};


%==========================================================================
% Run
%==========================================================================
function out = run_orthogonalise(job)
S = struct('D',{cellstr(job.D)},'chantype',{cellstr(job.chantype)},'prefix',job.prefix);
out.D = meeg_names(spm_osld_orthogonalise(S));


function out = run_sign_flip(job)
S = struct('D',{cellstr(job.D)},'chantype',{cellstr(job.chantype)},'nembed',job.nembed, ...
    'twostage',job.twostage,'nrandom',job.nrandom,'prefix',job.prefix);
template = char(job.template);
if ~isempty(template), S.template = template; end
out.D = meeg_names(spm_osld_sign_flip(S));


function out = run_hmm(job)
S = struct('D',{cellstr(job.D)},'chantype',{cellstr(job.chantype)},'K',job.K, ...
    'standardise',job.standardise,'hmm',job.training,'seed',job.seed, ...
    'prefix',job.prefix);
if job.pca > 0, S.pca = job.pca; end
if isfield(job.embedding,'tde')
    S.tde = struct('lags',-job.embedding.tde.lags:job.embedding.tde.lags, ...
        'npca',job.embedding.tde.npca);
end
switch job.means
    case 'learn', S.zeromean = false;
    case 'zero',  S.zeromean = true;
end
[HMM,Dout] = spm_osld_hmm(S);
HMM.data_files  = S.D(:);
HMM.state_files = meeg_names(Dout);
out.HMM = {fullfile(char(job.outdir),[job.name '.mat'])};
save(out.HMM{1},'HMM',spm_get_defaults('mat.format'));
out.D = HMM.state_files;


function dep = vout_hmm(job) %#ok<INUSD>
dep(1)            = cfg_dep;
dep(1).sname      = 'HMM file';
dep(1).src_output = substruct('.','HMM');
dep(1).tgt_spec   = cfg_findspec({{'filter','mat'}});
dep(2)            = meeg_dep('State probability datasets');
dep(2).src_output = substruct('.','D');


%==========================================================================
% Helpers
%==========================================================================
function f = meeg_files(tag,num)
f        = cfg_files;
f.tag    = tag;
f.name   = 'M/EEG datasets';
f.filter = 'mat';
f.num    = num;
f.help   = {'Select the M/EEG .mat files, e.g. one per session.'};


function c = chantype_entry
c         = cfg_entry;
c.tag     = 'chantype';
c.name    = 'Channel types';
c.help    = {['Types of the (good) channels to use, e.g. LFP for DAiSS parcel time ' ...
    'courses; one per line.']};
c.strtype = 's+';
c.num     = [1 Inf];
c.val     = {{'MEEG';'LFP';'SRC'}};


function p = prefix_entry(def)
p         = cfg_entry;
p.tag     = 'prefix';
p.name    = 'Output prefix';
p.help    = {'Prefix of the names of the datasets written.'};
p.strtype = 's';
p.num     = [1 Inf];
p.val     = {def};


function e = int_entry(tag,name,def,help)
e         = cfg_entry;
e.tag     = tag;
e.name    = name;
e.help    = {help};
e.strtype = 'n';
e.num     = [1 1];
e.val     = {def};


function e = real_entry(tag,name,def,help)
e         = cfg_entry;
e.tag     = tag;
e.name    = name;
e.help    = {help};
e.strtype = 'r';
e.num     = [1 1];
e.val     = {def};


function dep = meeg_dep(name)
dep            = cfg_dep;
dep.sname      = name;
dep.src_output = substruct('.','D');
dep.tgt_spec   = cfg_findspec({{'filter','mat'}});


function f = meeg_names(D)
if ~iscell(D), D = {D}; end
f = cellfun(@fullfile,D(:),'UniformOutput',false);
