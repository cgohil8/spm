function V = spm_osld_parcel_weights(C,w,method,pos,centres)
% Weights that combine source time courses into parcel time courses
% FORMAT V = spm_osld_parcel_weights(C,w,method,pos,centres)
%
% C       - [n x n] covariance of the n source time courses (normalised by
%           the number of samples), or a function handle C(ind) returning
%           the covariance of sources ind
% w       - [n x P] non-negative parcel weights of each source (from the
%           atlas)
% method  - 'pca' (default) or 'centroid'
% pos     - [n x 3] source coordinates in mm (needed for 'centroid')
% centres - [P x 3] parcel centres in mm for 'centroid' (e.g. the atlas
%           centres from spm_osld_parcel_maps)    [weighted centre of sources]
%
% V       - [n x P] such that parcel time courses are V' * (x - mean(x))
%
% 'pca', as osl_dynamics.meeg.parcellation._get_parcel_weights: for parcel p, with
% s = w(ind,p)/max(w(ind,p)) over its sources ind (weight > 0):
%   u     = leading eigenvector of diag(s)*C(ind,ind)*diag(s), eigenvalue d
%   m     = sources with s > 0.5 and non-zero variance (else all with
%           non-zero variance); sigma = sqrt(diag(C(ind,ind)))
%   sign  = sign(mean(u(m))),  scale = sum(|u(m)|.*sigma(m))/sum(|u(m)|)
%   V(ind,p) = sign*scale/sqrt(d) * u .* s
% so the parcel time course is the first principal component of the
% weighted sources, with standard deviation equal to the |u|-weighted mean
% of the source standard deviations and a positive mean loading.
% 'centroid', as osl-dynamics _get_parcel_data_centroid: the source nearest
% to the parcel centre, unscaled (with atlas centres, a parcel needs no
% source of its own).
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


if nargin < 3 || isempty(method), method = 'pca'; end
if isnumeric(C), Cmat = C; C = @(ind) Cmat(ind,ind); end
if any(w(:) < 0), error('spm_osld:invalidInput','Parcel weights must be non-negative.'); end
[n,P] = size(w);
V = zeros(n,P);
empty = [];
switch lower(method)
    case 'pca'
        for p = 1:P
            ind = find(w(:,p) > 0);
            if isempty(ind), empty(end+1) = p; continue; end %#ok<AGROW>
            s   = w(ind,p)/max(w(ind,p));
            Cp  = C(ind);
            sig = sqrt(max(diag(Cp),0));
            m   = s > 0.5 & sig > 0;
            if ~any(m), m = sig > 0; end
            if ~any(m), empty(end+1) = p; continue; end %#ok<AGROW>
            Cw  = (s*s').*Cp;
            Cw  = (Cw + Cw')/2;
            [U,d] = eig(Cw,'vector');
            [d,k] = max(d);
            u   = U(:,k);
            sd  = max(sqrt(abs(d)),eps);
            rw  = abs(u(m))/sum(abs(u(m)));
            V(ind,p) = sign(mean(u(m)))*(rw'*sig(m))/sd*(u.*s);
        end
    case 'centroid'
        if nargin < 4, error('spm_osld:invalidInput','pos is needed for the centroid method.'); end
        for p = 1:P
            if nargin > 4 && ~isempty(centres)
                c = centres(p,:);
            else
                if ~any(w(:,p) > 0), empty(end+1) = p; continue; end %#ok<AGROW>
                c = (w(:,p)'*pos)/sum(w(:,p));
            end
            [~,k] = min(sum((pos - c).^2,2));
            V(k,p) = 1;
        end
    otherwise
        error('spm_osld:invalidInput','Unknown method ''%s''.',method);
end
if ~isempty(empty)
    error('spm_osld:invalidInput','Parcel(s) %s contain no sources.',mat2str(empty));
end
