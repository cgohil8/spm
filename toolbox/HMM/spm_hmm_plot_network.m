function F = spm_hmm_plot_network(conn,atlas,S)
% Plot a network (e.g. thresholded coherence) on glass-brain views
% FORMAT F = spm_hmm_plot_network(conn,atlas,S)
%
% conn       - [P x P] network (zero for absent edges), e.g. one state of
%              spm_hmm_threshold_network / spm_hmm_gmm_threshold output
% atlas      - atlas NIfTI (parcel centres), or [P x 3] centres in mm
% S          - options:
%   .views     'x' (sagittal), 'y' (coronal), 'z' (axial), any
%              combination                                          ['xz']
%   .clim      [vmin vmax]                 [+/- max |edge value|, symmetric]
%   .mesh      mesh for the outline [canonical/cortex_8196.surf.gii]
%   .title     figure title                                            ['']
%
% F          - figure handle
%
% As osl-dynamics' connectivity.save: nodes at the parcel centres (black),
% an edge for every non-zero entry of the upper triangle, coloured on a
% symmetric red-blue scale. The glass brain is the silhouette and projection of the cortical
% mesh.
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


if nargin < 3, S = struct(); end
def = struct('views','xz','clim',[],'mesh',fullfile(spm('Dir'),'canonical','cortex_8196.surf.gii'),...
    'title','');
fn = fieldnames(def);
for i = 1:numel(fn)
    if ~isfield(S,fn{i}), S.(fn{i}) = def.(fn{i}); end
end
if ischar(atlas), c = spm_hmm_parcel_maps([],atlas); else, c = atlas; end
P = size(c,1);
if ~isequal(size(conn),[P P])
    error('spm_hmm:invalidInput','conn must be [P x P] with P = %d parcels.',P);
end
conn(1:P+1:end) = 0;
conn(isnan(conn)) = 0;
[i,j] = find(triu(conn ~= 0,1));
v = conn(sub2ind([P P],i,j));
cl = S.clim;
if isempty(cl), vmax = max(abs(v)); if isempty(vmax) || vmax == 0, vmax = 1; end, cl = [-vmax vmax]; end
cmap = red_blue(256);

M   = gifti(S.mesh);
vtx = double(M.vertices); fc = double(M.faces);
axs = struct('x',[2 3],'y',[1 3],'z',[1 2]);             % projected coordinates
F = figure('Color','w','Name','Network');
nv = numel(S.views);
for a = 1:nv
    d  = axs.(S.views(a));
    ax = subplot(1,nv,a); hold(ax,'on');
    x  = vtx(:,d(1)); y = vtx(:,d(2));
    b  = boundary(x,y,0.9);                              % silhouette
    fill(ax,x(b),y(b),[0.96 0.96 0.96],'EdgeColor','none','Tag','brain');
    patch(ax,'Faces',fc,'Vertices',vtx(:,d),'FaceColor',[0.4 0.4 0.4],...
        'FaceAlpha',0.02,'EdgeColor','none','Tag','brain');
    plot(ax,x(b),y(b),'-','Color',[0.45 0.45 0.45],'LineWidth',1,'Tag','brain');
    for e = 1:numel(v)
        k = 1 + round((v(e) - cl(1))/(cl(2) - cl(1))*255);
        k = min(max(k,1),256);
        plot(ax,c([i(e) j(e)],d(1)),c([i(e) j(e)],d(2)),'-','Color',cmap(k,:),...
            'LineWidth',2,'Tag','edge');
    end
    plot(ax,c(:,d(1)),c(:,d(2)),'k.','MarkerSize',10,'Tag','node');
    axis(ax,'equal','off');
    colormap(ax,cmap); caxis(ax,cl); %#ok<CAXIS> clim needs R2022a
end
colorbar(ax);
if ~isempty(S.title), sgtitle(F,S.title); end


%==========================================================================
function c = red_blue(n)
% blue - white - red (as matplotlib RdBu_r)
h = floor(n/2);
t = linspace(0,1,h)';
c = [[t t ones(h,1)]; [ones(h,1) flipud(t) flipud(t)]];
