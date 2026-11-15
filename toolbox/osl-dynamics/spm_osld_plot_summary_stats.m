function F = spm_osld_plot_summary_stats(stats,S)
% Plot summary statistics of the state time courses
% FORMAT F = spm_osld_plot_summary_stats(stats,S)
%
% stats           - HMM.stats of spm_osld_hmm: [G x K] fractional occupancy, mean
%                   lifetime, mean interval and switching rate of G
%                   sessions and K states
% S               - options:
%   .units          time unit of the lifetimes and intervals         ['s']
%   .title          figure title                                       ['']
%
% F               - figure handle
%
% As osl_dynamics.utils.plotting.plot_hmm_summary_stats: four panels with
% one violin per state over sessions (see plot_violins). States never
% visited in a session (NaN lifetime or interval) are left out of those
% violins.
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


if nargin < 2, S = struct(); end
def = struct('units','s','title','');
fn = fieldnames(def);
for i = 1:numel(fn)
    if ~isfield(S,fn{i}), S.(fn{i}) = def.(fn{i}); end
end
K   = size(stats.fo,2);
col = state_colours(K);
if strcmp(S.units,'s'), rate = 'Hz'; else, rate = 'per sample'; end
vals = {stats.fo, stats.mean_lifetime, stats.mean_interval, stats.switching_rate};
labs = {'Fractional occupancy', sprintf('Mean lifetime (%s)',S.units), ...
        sprintf('Mean interval (%s)',S.units), sprintf('Switching rate (%s)',rate)};

F = figure('Color','w','Name','Summary statistics','Position',[100 100 1200 300]);
for p = 1:4
    ax = subplot(1,4,p);
    plot_violins(ax,vals{p},col);
    xlabel(ax,'State'); ylabel(ax,labs{p});
    set(ax,'Tag',labs{p});
    box(ax,'off');
end
if ~isempty(S.title), sgtitle(F,S.title); end
