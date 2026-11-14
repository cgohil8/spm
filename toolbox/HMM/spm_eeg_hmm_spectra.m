function spec = spm_eeg_hmm_spectra(S)
% State-specific multitaper spectra of M/EEG datasets with osl-dynamics
% FORMAT spec = spm_eeg_hmm_spectra(S)
%
% S              - input structure:
%   .D             cell array of datasets (one per session) whose spectra
%                  are computed, e.g. the parcel data the HMM was fitted to
%                  (before embedding)
%   .HMM           fitted model from spm_eeg_hmm (or its model file)
%   .states        'viterbi' (one-hot Viterbi paths in S.HMM), or a cell
%                  array of the state probability datasets written by
%                  spm_eeg_hmm, one per dataset                  ['viterbi']
%   .channels      channel labels                [S.HMM.meeg.chanlabels]
%   .freq          [fmin fmax] frequency range in Hz       [0 to Nyquist]
%   .window        window length in samples            [2 x sampling rate]
%   .step          step between windows in samples             [.window]
%   .nw            time half-bandwidth                                 [4]
%   .ntapers       number of tapers                                    [7]
%   .standardise   z-score the data of each dataset first           [true]
%   .coh           also compute the coherence                       [true]
%   .cpsd          return cross spectra (.cpsd) instead of .psd    [false]
%   .python        Python executable                  [spm_osld_python]
%
% spec           - .f [F x 1] frequencies (Hz), .psd [G x K x D x F]
%                  (sessions, states, channels, frequencies) or .cpsd
%                  [G x K x D x D x F], .coh [G x K x D x D x F] (with
%                  .coh), .chanlabels
%
% As osl_dynamics.analysis.spectral.multitaper_spectra. The good samples of
% each dataset that have a state (not lost by time-delay embedding) are
% concatenated, as for spm_eeg_hmm, and the spectra are weighted by the
% state probabilities (or Viterbi paths).
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


def = struct('states','viterbi','channels',{{}},'freq',[],'window',[],'step',[], ...
    'nw',4,'ntapers',7,'standardise',true,'coh',true,'cpsd',false,'python','');
fn = fieldnames(def);
for i = 1:numel(fn)
    if ~isfield(S,fn{i}), S.(fn{i}) = def.(fn{i}); end
end
HMM = S.HMM;
if ischar(HMM), HMM = getfield(load(HMM),'HMM'); end
if isempty(S.channels), S.channels = HMM.meeg.chanlabels; end
D = S.D;
if ~iscell(D), D = {D}; end
K = HMM.K;

%-Data and state time courses on the same samples
%--------------------------------------------------------------------------
[X,info] = meeg_read(D,struct('channels',{S.channels}));
alpha = cell(numel(X),1);
for g = 1:numel(X)
    if ischar(S.states) && strcmpi(S.states,'viterbi')
        idx = HMM.meeg.index{g}(~HMM.missing{g});
        a = full(sparse(1:numel(idx),HMM.path{g}(~HMM.missing{g}),1,numel(idx),K));
    else
        Ds = S.states{g};
        if ~isa(Ds,'meeg'), Ds = spm_eeg_load(Ds); end
        a = reshape(Ds(1:K,:,:),K,[])';
        idx = find(all(isfinite(a),2));
        a = a(idx,:);
    end
    [~,ia,ib] = intersect(info.index{g},idx);
    X{g} = X{g}(ia,:);
    alpha{g} = single(a(ib,:));
end

%-Spectra
%--------------------------------------------------------------------------
args = struct('data',{X},'alpha',{alpha},'sampling_frequency',info.fsample, ...
    'window_length',S.window,'step_size',S.step,'time_half_bandwidth',S.nw, ...
    'n_tapers',S.ntapers,'frequency_range',S.freq,'standardize',logical(S.standardise), ...
    'calc_coh',logical(S.coh),'calc_cpsd',logical(S.cpsd));
r = osld_run('spectra',args,struct('python',S.python));
spec.f = r.f(:);
if S.cpsd, spec.cpsd = r.cpsd; else, spec.psd = r.psd; end
if S.coh, spec.coh = r.coh; end
spec.chanlabels = info.chanlabels;
