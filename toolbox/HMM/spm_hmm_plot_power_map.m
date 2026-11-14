function [F,vals] = spm_hmm_plot_power_map(values,atlas,S)
% Plot parcel maps (e.g. state power maps) on the cortical surface
% FORMAT [F,vals] = spm_hmm_plot_power_map(values,atlas,S)
%
% values          - [P x N] values of each parcel for N maps, e.g. the
%                   power of each state of session g and component c from
%                   spm_hmm_power_maps: reshape(P(g,c,:,:),K,[])'
% atlas           - atlas NIfTI used for the parcellation
% S               - options:
%   .subtractmean   subtract the (weighted) mean over the N maps    [false]
%   .meanweights    [N x 1] weights for that mean                [uniform]
%   .maps           indices of the maps to plot                    [1:N]
%   .clim           [vmin vmax]; default symmetric: +/- max |value| of
%                   each map                                          [[]]
%   .mesh           cortical mesh (GIfTI, MNI space)
%                   [canonical/cortex_8196.surf.gii]
%   .maxdist        vertices further than this (mm) from a parcel voxel
%                   are not coloured                  [1.5 x voxel size]
%   .titles         cell array with one title per plotted map    [none]
%
% F               - figure handles, one per map
% vals            - [n_vertices x numel(maps)] values shown at the vertices
%
% As osl-dynamics' plot_brain_surface: four views per map (left and right
% hemisphere, lateral and medial) on a symmetric cold-hot colour scale.
% Each vertex takes the value of the nearest atlas voxel with a non-zero
% weight.
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


if nargin < 3, S = struct(); end
N = size(values,2);
def = struct('subtractmean',false,'meanweights',[],'maps',1:N,'clim',[],...
    'mesh',fullfile(spm('Dir'),'canonical','cortex_8196.surf.gii'),...
    'maxdist',[],'titles',{{}});
fn = fieldnames(def);
for i = 1:numel(fn)
    if ~isfield(S,fn{i}), S.(fn{i}) = def.(fn{i}); end
end
if S.subtractmean && N > 1
    w = S.meanweights; if isempty(w), w = ones(N,1); end
    values = values - values*(w(:)/sum(w));
end

%-Values at the mesh vertices
%--------------------------------------------------------------------------
A = read_atlas(atlas);
if size(values,1) ~= A.P
    error('spm_hmm:invalidInput','The atlas has %d parcels, values has %d rows.',A.P,size(values,1));
end
M   = gifti(S.mesh);
vtx = double(M.vertices);
fc  = double(M.faces);
inparc = find(any(A.W,2));
[i,j,k] = ind2sub(A.dim,inparc);
xyz = [i j k ones(numel(i),1)]*A.mat';
xyz = xyz(:,1:3);
if isempty(S.maxdist), S.maxdist = 1.5*mean(sqrt(sum(A.mat(1:3,1:3).^2,1))); end
[near,dist] = nearest_point(vtx,xyz);
mx  = full(max(A.W,[],1)); mx(mx == 0) = 1;
Wv  = full(A.W(inparc(near),:))./mx;
vals = Wv*values(:,S.maps);
vals(dist > S.maxdist,:) = NaN;

%-Plot
%--------------------------------------------------------------------------
lf = all(reshape(vtx(fc(:),1) < 0,size(fc)),2);
rf = all(reshape(vtx(fc(:),1) >= 0,size(fc)),2);
views = {lf,-90,'Left lateral'; rf,90,'Right lateral'; lf,90,'Left medial'; rf,-90,'Right medial'};
F = gobjects(numel(S.maps),1);
for m = 1:numel(S.maps)
    v = vals(:,m);
    cl = S.clim;
    if isempty(cl), vmax = max(abs(v),[],'omitnan'); if vmax == 0 || isnan(vmax), vmax = 1; end, cl = [-vmax vmax]; end
    F(m) = figure('Color','w','Name',sprintf('Map %d',S.maps(m)));
    for p = 1:4
        ax = subplot(2,2,p);
        patch(ax,'Faces',fc(views{p,1},:),'Vertices',vtx,'FaceVertexCData',v,...
            'FaceColor','interp','EdgeColor','none','FaceLighting','gouraud',...
            'AmbientStrength',0.4,'DiffuseStrength',0.7,'SpecularStrength',0.1);
        u = vtx(unique(fc(views{p,1},:)),:);           % limits before camlight,
        axis(ax,'equal','off'); view(ax,views{p,2},0);  % which would extend them
        set(ax,'XLim',[min(u(:,1)) max(u(:,1))],'YLim',[min(u(:,2)) max(u(:,2))],...
            'ZLim',[min(u(:,3)) max(u(:,3))]);
        camlight(ax,'headlight');
        set(ax,'Color',[0.7 0.7 0.7]); colormap(ax,cold_hot(256));
        caxis(ax,cl); %#ok<CAXIS> clim needs R2022a
        title(ax,views{p,3});
    end
    colorbar(ax,'Position',[0.92 0.3 0.02 0.4]);
    if ~isempty(S.titles), sgtitle(F(m),S.titles{m}); end
end


%==========================================================================
function [idx,d] = nearest_point(a,b)
% nearest row of b for each row of a (Euclidean), in blocks
idx = zeros(size(a,1),1); d = idx;
bb  = sum(b.^2,2)';
nb  = max(1,floor(5e6/size(b,1)));                    % ~40 MB per block
for lo = 1:nb:size(a,1)
    hi = min(lo+nb-1,size(a,1));
    D2 = sum(a(lo:hi,:).^2,2) + bb - 2*a(lo:hi,:)*b';
    [m,idx(lo:hi)] = min(D2,[],2);
    d(lo:hi) = sqrt(max(m,0));
end

%==========================================================================
function c = cold_hot(n)
% cyan - blue - black - red - yellow (as nilearn's cold_hot)
h = floor(n/2);
t = linspace(0,1,h)';
neg = [zeros(h,1), max(1-2*t,0), min(1,2-2*t)];          % cyan -> blue -> black
pos = [min(1,2*t), max(2*t-1,0), zeros(h,1)];             % black -> red -> yellow
c = [neg; pos];
