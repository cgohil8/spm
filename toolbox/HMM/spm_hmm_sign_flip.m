function [flips,info] = spm_hmm_sign_flip(Y,S)
% Align the signs of channels across sessions
% FORMAT [flips,info] = spm_hmm_sign_flip(Y,S)
%
% Y              - data: cell array of [T_i x D] segments, or a data reader
%                  (rows with NaN are missing)
% S              - options:
%   .group         [nseg x 1] session of each segment  [each segment its own]
%   .nembed        number of embeddings (odd)                           [15]
%   .standardise   standardise embedded data before the covariance    [true]
%   .template      template covariance (from spm_hmm_embedded_cov or
%                  spm_hmm_build_template), or [] to use the median session
%                  (or the dataset template, if .twostage)              [[]]
%   .twostage      build a template from these sessions, align it to
%                  .template, and align each session to it (see below)
%                                                                     [false]
%   .nrandom       random starts of each flip search                     [3]
%   .bands, .fs    covariances per frequency band (spm_hmm_embedded_cov)
%   .missing       cell array of missing-sample masks per segment       [{}]
%
% flips          - [n_sessions x D] signs (+1/-1); multiply each segment
%                  of session g by flips(g,:)
% info           - .reader       data reader returning the flipped data
%                  .corr         [n_sessions x 1] correlation of each
%                                session with the template it was aligned
%                                to (before .template_flips)
%                  .template     the template the sessions were aligned to
%                  .template_session  median session (default template)
%                  .medoid       medoid session (.twostage)
%                  .template_flips    flips aligning the dataset template
%                                to S.template (.twostage)
%                  .covs         session covariances
%
% As osl_dynamics.data.Data.align_channel_signs:
% - no template: the template is the covariance of the session with the
%   median summed sign-invariant similarity to the other sessions;
% - with a template: every session is aligned to it;
% - twostage: a dataset template is built (spm_hmm_build_template); with
%   S.template, the flips g aligning the dataset template to S.template
%   are found; every session is aligned to the dataset template and its
%   flips multiplied by g. For an S.template from another dataset (e.g. a
%   pre-trained model).
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


R = data_reader(Y);
if nargin < 2, S = struct(); end
def = struct('group',[],'nembed',15,'standardise',true,'template',[],...
    'twostage',false,'nrandom',3,'bands',[],'fs',[],'missing',{{}});
fn = fieldnames(def);
for i = 1:numel(fn)
    if ~isfield(S,fn{i}), S.(fn{i}) = def.(fn{i}); end
end
if isempty(S.group), S.group = (1:R.nseg)'; end
[~,~,group] = unique(S.group(:));
if numel(group) ~= R.nseg
    error('spm_hmm:invalidInput','S.group must have one entry per segment.');
end
G = max(group);
E = S.nembed;

%-Covariance of each session
%--------------------------------------------------------------------------
covs = cell(G,1);
for g = 1:G
    seg = find(group == g)';
    Sc  = struct('nembed',E,'standardise',S.standardise,'segments',seg,...
        'bands',S.bands,'fs',S.fs);
    if ~isempty(S.missing), Sc.missing = S.missing(seg); end
    covs{g} = spm_hmm_embedded_cov(R,Sc);
end

%-Template
%--------------------------------------------------------------------------
template = S.template;
info.template_session = [];
info.medoid           = [];
info.template_flips   = [];
tflips = ones(1,R.D);
if S.twostage
    [dataset_template,info.medoid] = spm_hmm_build_template(covs,E,2,S.nrandom);
    if ~isempty(template)
        tflips = spm_hmm_find_flips(dataset_template,template,E,S.nrandom);
        info.template_flips = tflips;
    end
    template = dataset_template;
elseif isempty(template)
    msum = sum(sign_invariant_similarity(covs,E),2);
    [~,o] = sort(msum);
    info.template_session = o(floor(G/2) + 1);
    template = covs{info.template_session};
end
if ~isequal(size(template),size(covs{1}))
    error('spm_hmm:invalidInput','S.template must match the size of the session covariances.');
end

%-Flips
%--------------------------------------------------------------------------
flips     = zeros(G,R.D);
info.corr = zeros(G,1);
for g = 1:G
    [f,info.corr(g)] = spm_hmm_find_flips(covs{g},template,E,S.nrandom);
    flips(g,:) = f.*tflips;
end

info.template = template;
info.covs     = covs;
info.reader   = R;
info.reader.read = @(i,lo,hi) R.read(i,lo,hi).*flips(group(i),:);
