function [conn,edges] = spm_osld_threshold_network(conn,S)
% Keep the strongest edges of connectivity networks
% FORMAT [conn,edges] = spm_osld_threshold_network(conn,S)
%
% conn            - [C x K x D x D] networks (components x states),
%                   [K x D x D], [D x D], or [G x C x K x D x D] (sessions,
%                   as from spm_osld_coherence_networks; each session is
%                   thresholded separately)
% S               - options:
%   .percentile     percentile threshold: scalar or [C x K]       [required]
%   .subtractmean   subtract the (weighted) mean over states first  [false]
%   .meanweights    [K x 1] weights for that mean                [uniform]
%   .absolute       threshold the absolute value                    [false]
%   .python         Python executable                 [spm_osld_python]
%
% conn            - input with the edges that are not kept set to zero
% edges           - logical, true for the edges kept
%
% As osl_dynamics.analysis.connectivity.threshold.
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


def = struct('subtractmean',false,'meanweights',[],'absolute',false,'python','');
fn = fieldnames(def);
for i = 1:numel(fn)
    if ~isfield(S,fn{i}), S.(fn{i}) = def.(fn{i}); end
end
if ~isfield(S,'percentile') || isempty(S.percentile)
    error('spm_osld:invalidInput','S.percentile is required.');
end
sz = size(conn);
nd = ndims(conn);
switch nd
    case 2, c = reshape(conn,[1 1 sz]);  G = 1;
    case 3, c = reshape(conn,[1 sz]);    G = 1;
    case 4, c = conn;                    G = 1;
    case 5, G = sz(1);
    otherwise, error('spm_osld:invalidInput','conn must have 2 to 5 dimensions.');
end
if nd == 5
    edges = false(sz);
    for g = 1:G
        e = threshold(reshape(conn(g,:,:,:,:),sz(2:5)),S);
        edges(g,:,:,:,:) = reshape(e,[1 sz(2:5)]);
    end
else
    edges = reshape(threshold(c,S),sz);
end
conn(~edges) = 0;


function e = threshold(c,S)
args = struct('conn',c,'percentile',S.percentile,'subtract_mean',logical(S.subtractmean), ...
    'mean_weights',S.meanweights,'absolute_value',logical(S.absolute));
r = osld_run('threshold',args,struct('python',S.python));
e = reshape(r.edges,size(c)) > 0;
