function hmm_maths_check
% Numerical checks of the formulae in the VB-HMM maths note.
% Parameterisation: Wishart W(Lambda | W, nu) with E[Lambda] = nu*W (W = scale).
% Normal-Wishart: mu | Lambda ~ N(m, (kappa*Lambda)^-1).

rng(1);
fprintf('\n=== A. Wishart / Normal-Wishart / Dirichlet identities (Monte Carlo) ===\n');
D = 4; nS = 2e5;
W  = randspd(D);  nu  = D + 3.7;
W0 = randspd(D);  nu0 = D + 1.2;
m  = randn(D,1);  kap  = 3.1;
m0 = randn(D,1);  kap0 = 0.7;

Lam = wishrnd_(W,nu,nS);
ld  = zeros(nS,1); for s=1:nS, ld(s) = logdet_(Lam(:,:,s)); end
report('E[log|Lambda|]', mean(ld), Elogdet(nu,W), std(ld)/sqrt(nS));

lq = zeros(nS,1); lp = lq;
for s=1:nS, lq(s)=wishlogpdf(Lam(:,:,s),W,nu); lp(s)=wishlogpdf(Lam(:,:,s),W0,nu0); end
report('KL Wishart', mean(lq-lp), klW(nu,W,nu0,W0), std(lq-lp)/sqrt(nS));
report('spm_kl_wishart (inverse-scale args)', spm_kl_wishart(nu,inv(W),nu0,inv(W0)), klW(nu,W,nu0,W0), 0);
% spm_kl_wishart with E_q[log|L|] in place of E_p[log|L|] in the cross-entropy term
Lq = Elogdet(nu,W); Lp = Elogdet(nu0,W0);
report('spm_kl_wishart + 0.5(nu0-D-1)(Lp-Lq)', spm_kl_wishart(nu,inv(W),nu0,inv(W0)) + 0.5*(nu0-D-1)*(Lp-Lq), klW(nu,W,nu0,W0), 0);

lq = zeros(nS,1); lp = lq; y = randn(D,1); ll = lq;
for s=1:nS
    L  = Lam(:,:,s);
    mu = m + chol(kap*L)\randn(D,1);
    lq(s) = wishlogpdf(L,W,nu)   + mvnlogpdf_prec(mu,m ,kap *L);
    lp(s) = wishlogpdf(L,W0,nu0) + mvnlogpdf_prec(mu,m0,kap0*L);
    ll(s) = mvnlogpdf_prec(y,mu,L);
end
report('KL Normal-Wishart', mean(lq-lp), klNW(m,kap,W,nu,m0,kap0,W0,nu0), std(lq-lp)/sqrt(nS));
report('E_q[log N(y|mu,Lambda^-1)]', mean(ll), Eloglik(y',m,kap,W,nu), std(ll)/sqrt(nS));

a = rand(1,5)*3+0.2; a0 = rand(1,5)*2+0.5;
P = dirrnd_(a,nS);
lq = dirlogpdf(P,a); lp = dirlogpdf(P,a0);
report('KL Dirichlet', mean(lq-lp), klDir(a,a0), std(lq-lp)/sqrt(nS));
report('spm_kl_dirichlet', spm_kl_dirichlet(a,a0), klDir(a,a0), 0);
lP = log(P);
report('E[log pi_k] (k=2)', mean(lP(:,2)), psi(a(2))-psi(sum(a)), std(lP(:,2))/sqrt(nS));

fprintf('\n=== B. HMM: forward-backward vs brute-force enumeration, and F identity ===\n');
K = 3; D = 2; T = 6;
Y  = randn(T,D);
q  = randpost(K,D,T);  pr = defaultprior(K,D);
[lpi,lA,lb] = tilde(Y,q);
[g,Xi,logZ,xis] = fb(lpi,lA,lb);
[gB,XiB,logZB,Hb,EsB] = brute(lpi,lA,lb);
report('log Z~ (fb vs brute)', logZ, logZB, 0);
report('max|gamma diff|', max(abs(g(:)-gB(:))), 0, 0);
report('max|sum xi diff|', max(abs(Xi(:)-XiB(:))), 0, 0);
H = logZ - (g(1,:)*lpi + sum(sum(Xi.*lA)) + sum(sum(g.*lb)));
report('H[q(s)] via logZ - E[score] vs brute', H, Hb, 0);
report('chain-entropy formula vs brute', chainH(g,xis), Hb, 0);
F1 = logZ - KLtheta(q,pr);
F2 = EsB + Hb - KLtheta(q,pr);
report('F identity vs explicit E[score]+H-KL', F1, F2, 0);
% Monte-Carlo check that E_q(theta)[log p(Y,s|theta)] = score(s) for one path
s  = [1 2 2 3 1 1]; nS = 1e5; acc = zeros(nS,1);
th = sampletheta(q,nS);
for i=1:nS
    v = log(th.pi(i,s(1)));
    for t=2:T, v = v + log(th.A(s(t-1),s(t),i)); end
    for t=1:T, v = v + mvnlogpdf_prec(Y(t,:)',th.mu(:,s(t),i),th.Lam(:,:,s(t),i)); end
    acc(i)=v;
end
sc = lpi(s(1)) + sum(lA(sub2ind([K K],s(1:end-1),s(2:end)))) + sum(lb(sub2ind([T K],1:T,s)));
report('E_q(theta)[log p(Y,s|theta)] vs score(s)', mean(acc), sc, std(acc)/sqrt(nS));

fprintf('\n=== C. K=1: F equals exact log evidence ===\n');
D = 3; N = 40; Y = randn(N,D)*randspd(D) + 2;
for zm = [0 1]
    pr = defaultprior(1,D); pr.zeromean = zm;
    if zm, pr.m0 = zeros(D,1); end
    st = suffstats(Y,ones(N,1),zeros(1,1),ones(1,1));
    q  = mstep(st,pr);
    [lpi,lA,lb] = tilde(Y,q);
    [~,~,logZ] = fb(lpi,lA,lb);
    F = logZ - KLtheta(q,pr);
    report(sprintf('F vs log evidence (zeromean=%d)',zm), F, logev(Y,pr), 0);
end

fprintf('\n=== D. Batch VB: F monotone after every E- and M-step (general F) ===\n');
for zm = [0 1]
    [Y,segs] = simhmm(3,4,[1500 1000 700],zm);
    pr = defaultprior(3,4); pr.zeromean = zm; pr.m0 = zm*0 + (1-zm)*mean(Y)'; if zm, pr.m0=zeros(4,1); end
    % initialise: M-step from a random one-hot path
    g = zeros(size(Y,1),3); g(sub2ind(size(g),(1:size(Y,1))',randi(3,size(Y,1),1))) = 1;
    st = suffstats(Y,g,zeros(3),zeros(1,3)); st.Xi = ones(3); st.g1 = ones(1,3);
    q = mstep(st,pr);
    Fh = []; worst = inf; Hs = []; st = [];
    for it = 1:60
        % E-step
        [st,H] = estep(Y,segs,q);
        Fe = Fgeneral(st,H,q,pr);
        % identity check right after E-step
        Fid = st.logZ - KLtheta(q,pr);
        if it==1, report('F identity == general F after E-step', Fe, Fid, 0); end
        q = mstep(st,pr);
        Fm = Fgeneral(st,H,q,pr);
        Fh = [Fh Fe Fm]; %#ok<AGROW>
    end
    dF = diff(Fh);
    fprintf('  zeromean=%d: F %.4f -> %.4f, min step %.3e (should be >= -1e-9*|F|)\n', zm, Fh(1), Fh(end), min(dF));
end

fprintf('\n=== E. Padding / masking ===\n');
K = 3; D = 2; T = 40; Y = randn(T,D); q = randpost(K,D,T);
[lpi,lA,lb] = tilde(Y,q);
[g0,~,logZ0] = fb(lpi,lA,lb);
P = 15; lbp = [lb; zeros(P,K)];
[g1,~,logZ1] = fb(lpi,lA,lbp);
fprintf('  pad with log b~=0 and normal A~ : max|dgamma| = %.3e, dlogZ = %.3e  (NOT neutral)\n', max(max(abs(g1(1:T,:)-g0))), logZ1-logZ0);
[g2,~,logZ2] = fb_tv(lpi,lA,lbp,T);
fprintf('  pad with identity transitions   : max|dgamma| = %.3e, dlogZ = %.3e  (neutral)\n', max(max(abs(g2(1:T,:)-g0))), logZ2-logZ0);

fprintf('\n=== F. Chunking with buffers vs full forward-backward ===\n');
[Y,segs,qtrue] = simhmm(4,5,20000,0); %#ok<ASGLU>
pr = defaultprior(4,5); pr.m0 = mean(Y)';
q = qtrue; % posterior from the true states
[lpi,lA,lb] = tilde(Y,q);
[gfull,Xfull,~,xisfull] = fb(lpi,lA,lb); %#ok<ASGLU>
Ahat = exp(lA); Ahat = Ahat./sum(Ahat,2); [V,E] = eig(Ahat'); [~,i]=max(real(diag(E))); pst = real(V(:,i))'/sum(real(V(:,i)));
for buf = [0 10 50 200]
    Lc = 1000; gc = zeros(size(gfull)); Xc = zeros(4); G1 = zeros(1,4);
    for c0 = 1:Lc:size(Y,1)
        a = c0; b = min(c0+Lc-1,size(Y,1));
        lo = max(1,a-buf); hi = min(size(Y,1),b+buf);
        if lo==1, lp0 = lpi; else, lp0 = log(pst)'; end
        [gg,~,~,xx] = fb(lp0,lA,lb(lo:hi,:));
        gc(a:b,:) = gg(a-lo+1:b-lo+1,:);
        % ownership: chunk owns xi into t for t in [a,b], t>1
        tt = (max(a,2):b) - lo + 1;      % xis index k corresponds to transition (k)->(k+1)
        tt = tt(tt>=2);                  % transition into first sample is only seen if buffer>=1
        Xc = Xc + squeeze(sum(xx(:,:,tt-1),3));
        if a==1, G1 = G1 + gg(1,:); end
    end
    fprintf('  buffer=%4d: max|dgamma|=%.2e, max|dXi|/sum=%.2e, sum gamma diff=%.2e\n', buf, ...
        max(abs(gc(:)-gfull(:))), max(abs(Xc(:)-Xfull(:)))/sum(Xfull(:)), abs(sum(gc(:))-sum(gfull(:))));
end

fprintf('\n=== G. Sticky transition prior: lifetime L -> delta ===\n');
for K = [6 8 12], for L = [5 10 25]
    b0 = 1; dl = b0*(K*(L-1)-L);
    p = (b0+dl)/(K*b0+dl);
    fprintf('  K=%2d L=%3d: delta=%6.1f, prior E[A_ii]=%.4f, 1/(1-E[A_ii])=%.3f\n',K,L,dl,p,1/(1-p));
end, end

fprintf('\n=== H. SVI: rho=1, full batch, chunks=segments == batch VB update ===\n');
[Y,segs] = simhmm(3,3,[800 600],0);
pr = defaultprior(3,3); pr.m0 = mean(Y)';
q = mstep(suffstats(Y,onehot(randi(3,size(Y,1),1),3),ones(3),ones(1,3)),pr);
st = estep(Y,segs,q);
qb = mstep(st,pr);
eta  = natparams(q,pr);             % current
etab = natparams(qb,pr);
eta0 = natparams_prior(pr);
etaS = blend(eta, addeta(eta0, statseta(st,1)), 1);
fprintf('  max |eta_svi - eta_batch| = %.3e\n', maxdiff(etaS,etab));
% unbiasedness of c = M/|B| over random batches of segments (unequal lengths)
[Y,segs] = simhmm(3,3,[300 900 500 1200 400 700],0);
stAll = estep(Y,segs,q); Mseg = size(segs,1); Bsz = 2; acc = 0; nrep = 4000;
for r=1:nrep
    idx = randperm(Mseg,Bsz);
    sb = estep(Y,segs(idx,:),q);
    acc = acc + (Mseg/Bsz)*[sb.N(:); sb.Xi(:); sb.g1(:)];
end
ref = [stAll.N(:); stAll.Xi(:); stAll.g1(:)];
fprintf('  c=M/|B|: max rel bias = %.3e (MC noise ~1e-2)\n', max(abs(acc/nrep-ref)./ref));
fprintf('\nDone.\n');
end

%==========================================================================
function report(name,a,b,se)
if se>0
    fprintf('  %-48s %+.6f vs %+.6f  (diff %.2e, %.1f SE)\n',name,a,b,a-b,abs(a-b)/se);
else
    fprintf('  %-48s %+.6g vs %+.6g  (diff %.2e)\n',name,a,b,a-b);
end
end

function S = randspd(D), X = randn(D,2*D); S = X*X'/(2*D) + 0.3*eye(D); end
function v = logdet_(A), v = 2*sum(log(diag(chol(A)))); end
function v = psiD(a,D), v = sum(psi(a + (1-(1:D))/2)); end
function v = lgD(a,D), v = D*(D-1)/4*log(pi) + sum(gammaln(a + (1-(1:D))/2)); end
function v = Elogdet(nu,W), D = size(W,1); v = psiD(nu/2,D) + D*log(2) + logdet_(W); end

function v = klW(nu,W,nu0,W0)
D = size(W,1);
v = nu0/2*(logdet_(W0)-logdet_(W)) + nu/2*(trace(W0\W)-D) + lgD(nu0/2,D) - lgD(nu/2,D) + (nu-nu0)/2*psiD(nu/2,D);
end
function v = klNW(m,kap,W,nu,m0,kap0,W0,nu0)
D = size(W,1); d = m-m0;
v = klW(nu,W,nu0,W0) + 0.5*(D*(kap0/kap - 1 - log(kap0/kap)) + kap0*nu*(d'*W*d));
end
function v = klDir(a,a0)
v = gammaln(sum(a)) - sum(gammaln(a)) - gammaln(sum(a0)) + sum(gammaln(a0)) + sum((a-a0).*(psi(a)-psi(sum(a))));
end
function v = wishlogpdf(L,W,nu)
D = size(W,1);
v = (nu-D-1)/2*logdet_(L) - 0.5*trace(W\L) - nu*D/2*log(2) - nu/2*logdet_(W) - lgD(nu/2,D);
end
function v = mvnlogpdf_prec(x,mu,P)
D = numel(x); d = x-mu; v = -D/2*log(2*pi) + 0.5*logdet_(P) - 0.5*(d'*P*d);
end
function v = dirlogpdf(P,a)
v = gammaln(sum(a)) - sum(gammaln(a)) + log(P)*(a(:)-1);
end
function L = wishrnd_(W,nu,n)
D = size(W,1); C = chol(W,'lower'); L = zeros(D,D,n);
for s=1:n
    A = tril(randn(D),-1);
    A(1:D+1:end) = sqrt(2*randg((nu-(1:D)+1)/2));
    B = C*A; L(:,:,s) = B*B';
end
end
function P = dirrnd_(a,n), G = randg(repmat(a,n,1)); P = G./sum(G,2); end

function v = Eloglik(Y,m,kap,W,nu)
% E_q[log N(y | mu, Lambda^-1)] for rows of Y (T x D); m=[] -> zero mean
D = size(W,1); C = chol(W);  % W = C'*C
if isempty(m), d = Y; corr = 0; else, d = Y - m'; corr = D/kap; end
Z = d*C';
v = -D/2*log(2*pi) + 0.5*Elogdet(nu,W) - 0.5*(corr + nu*sum(Z.^2,2));
end

%--------------------------------------------------------------------------
function pr = defaultprior(K,D)
pr.K = K; pr.D = D; pr.zeromean = 0;
pr.alpha0 = ones(1,K); pr.beta0 = ones(K) + 5*eye(K);
pr.m0 = zeros(D,1); pr.kap0 = 1; pr.nu0 = D+2; pr.W0 = inv((pr.nu0-D-1)*eye(D));
end
function q = randpost(K,D,T)
q.zeromean = 0;
q.a  = rand(1,K)*2+0.5; q.B = rand(K)*3+0.3 + 4*eye(K);
for k=1:K, q.m(:,k)=randn(D,1); q.kap(k)=rand*5+0.5; q.nu(k)=D+1+rand*5; q.W(:,:,k)=randspd(D)/q.nu(k); end
end
function [lpi,lA,lb] = tilde(Y,q)
K = numel(q.a);
lpi = (psi(q.a) - psi(sum(q.a)))';
lA  = psi(q.B) - psi(sum(q.B,2));
lb  = zeros(size(Y,1),K);
for k=1:K
    if q.zeromean, lb(:,k) = Eloglik(Y,[],[],q.W(:,:,k),q.nu(k));
    else,          lb(:,k) = Eloglik(Y,q.m(:,k),q.kap(k),q.W(:,:,k),q.nu(k)); end
end
end
function [g,Xi,logZ,xis] = fb(lpi,lA,lb)
[T,K] = size(lb); A = exp(lA);
mx = max(lb,[],2); B = exp(lb-mx);
al = zeros(T,K); be = ones(T,K); c = zeros(T,1);
al(1,:) = exp(lpi').*B(1,:); c(1) = sum(al(1,:)); al(1,:) = al(1,:)/c(1);
for t=2:T, al(t,:) = (al(t-1,:)*A).*B(t,:); c(t)=sum(al(t,:)); al(t,:)=al(t,:)/c(t); end
for t=T-1:-1:1, be(t,:) = ((B(t+1,:).*be(t+1,:))*A')/c(t+1); end
g = al.*be; xis = zeros(K,K,max(T-1,0));
for t=2:T, xis(:,:,t-1) = (al(t-1,:)'*(B(t,:).*be(t,:))).*A/c(t); end
Xi = sum(xis,3); logZ = sum(log(c)) + sum(mx);
end
function [g,Xi,logZ] = fb_tv(lpi,lA,lb,Treal)
% time-varying transitions: identity in padded region
[T,K] = size(lb); mx = max(lb,[],2); B = exp(lb-mx);
al = zeros(T,K); be = ones(T,K); c = zeros(T,1);
At = @(t) (t<=Treal)*exp(lA) + (t>Treal)*eye(K);
al(1,:) = exp(lpi').*B(1,:); c(1)=sum(al(1,:)); al(1,:)=al(1,:)/c(1);
for t=2:T, al(t,:)=(al(t-1,:)*At(t)).*B(t,:); c(t)=sum(al(t,:)); al(t,:)=al(t,:)/c(t); end
for t=T-1:-1:1, be(t,:)=((B(t+1,:).*be(t+1,:))*At(t+1)')/c(t+1); end
g = al.*be; Xi = []; logZ = sum(log(c))+sum(mx);
end
function [g,Xi,logZ,H,Es] = brute(lpi,lA,lb)
[T,K] = size(lb); n = K^T; S = zeros(n,T); sc = zeros(n,1);
for i=1:n
    s = 1 + mod(floor((i-1)./K.^(0:T-1)),K); S(i,:) = s;
    sc(i) = lpi(s(1)) + sum(lA(sub2ind([K K],s(1:end-1),s(2:end)))) + sum(lb(sub2ind([T K],1:T,s)));
end
logZ = max(sc) + log(sum(exp(sc-max(sc)))); p = exp(sc-logZ);
g = zeros(T,K); Xi = zeros(K);
for i=1:n
    s = S(i,:); g(sub2ind([T K],1:T,s)) = g(sub2ind([T K],1:T,s)) + p(i);
    for t=2:T, Xi(s(t-1),s(t)) = Xi(s(t-1),s(t)) + p(i); end
end
H = -sum(p.*log(p)); Es = sum(p.*sc);
end
function H = chainH(g,xis)
H = -sum(g(1,:).*log(g(1,:)));
for t=1:size(xis,3)
    x = xis(:,:,t); cond = x./g(t,:)';
    H = H - sum(x(x>0).*log(cond(x>0)));
end
end
function v = KLtheta(q,pr)
v = klDir(q.a,pr.alpha0);
for i=1:numel(q.a), v = v + klDir(q.B(i,:),pr.beta0(i,:)); end
for k=1:numel(q.a)
    if q.zeromean, v = v + klW(q.nu(k),q.W(:,:,k),pr.nu0,pr.W0);
    else, v = v + klNW(q.m(:,k),q.kap(k),q.W(:,:,k),q.nu(k),pr.m0,pr.kap0,pr.W0,pr.nu0); end
end
end
function th = sampletheta(q,n)
K = numel(q.a); D = size(q.W,1);
th.pi = dirrnd_(q.a,n); th.A = zeros(K,K,n);
for i=1:K, th.A(i,:,:) = permute(dirrnd_(q.B(i,:),n),[3 2 1]); end
th.mu = zeros(D,K,n); th.Lam = zeros(D,D,K,n);
for k=1:K
    L = wishrnd_(q.W(:,:,k),q.nu(k),n);
    for i=1:n, th.Lam(:,:,k,i)=L(:,:,i); th.mu(:,k,i) = q.m(:,k) + chol(q.kap(k)*L(:,:,i))\randn(D,1); end
end
end

%--------------------------------------------------------------------------
function st = suffstats(Y,g,Xi,g1)
K = size(g,2); D = size(Y,2);
st.N = sum(g,1); st.S1 = Y'*g; st.S2 = zeros(D,D,K);
for k=1:K, st.S2(:,:,k) = (Y.*g(:,k))'*Y; end
st.Xi = Xi; st.g1 = g1; st.logZ = NaN;
end
function q = mstep(st,pr)
K = numel(st.N); D = size(st.S1,1);
q.zeromean = pr.zeromean;
q.a = pr.alpha0 + st.g1; q.B = pr.beta0 + st.Xi;
for k=1:K
    N = st.N(k); q.nu(k) = pr.nu0 + N;
    if pr.zeromean
        Wi = inv(pr.W0) + st.S2(:,:,k); q.m(:,k) = zeros(D,1); q.kap(k) = Inf;
    else
        q.kap(k) = pr.kap0 + N;
        q.m(:,k) = (pr.kap0*pr.m0 + st.S1(:,k))/q.kap(k);
        if N > 0, xb = st.S1(:,k)/N; Sk = st.S2(:,:,k) - N*(xb*xb'); else, xb = zeros(D,1); Sk = zeros(D); end
        d = xb - pr.m0;
        Wi = inv(pr.W0) + Sk + (pr.kap0*N/(pr.kap0+N))*(d*d');
    end
    Wi = (Wi+Wi')/2; q.W(:,:,k) = inv(Wi); q.W(:,:,k) = (q.W(:,:,k)+q.W(:,:,k)')/2;
end
end
function [st,H] = estep(Y,segs,q)
K = numel(q.a); D = size(Y,2);
st.N = zeros(1,K); st.S1 = zeros(D,K); st.S2 = zeros(D,D,K); st.Xi = zeros(K); st.g1 = zeros(1,K); st.logZ = 0; H = 0;
[lpi,lA,lbAll] = tilde(Y,q);
for i=1:size(segs,1)
    idx = segs(i,1):segs(i,2); lb = lbAll(idx,:);
    [g,Xi,logZ] = fb(lpi,lA,lb);
    s = suffstats(Y(idx,:),g,Xi,g(1,:));
    st.N = st.N+s.N; st.S1 = st.S1+s.S1; st.S2 = st.S2+s.S2; st.Xi = st.Xi+Xi; st.g1 = st.g1+g(1,:);
    st.logZ = st.logZ + logZ;
    H = H + logZ - (g(1,:)*lpi + sum(sum(Xi.*lA)) + sum(sum(g.*lb)));
end
end
function F = Fgeneral(st,H,q,pr)
% F = E_q[log p(Y,s|theta)] + H[q(s)] - KL(q(theta)||p(theta)), using sufficient statistics
K = numel(q.a); D = size(st.S1,1);
lpi = (psi(q.a)-psi(sum(q.a)))'; lA = psi(q.B) - psi(sum(q.B,2));
E = st.g1*lpi + sum(sum(st.Xi.*lA));
for k=1:K
    W = q.W(:,:,k); N = st.N(k);
    if q.zeromean
        quad = trace(W*st.S2(:,:,k)); corr = 0;
    else
        m = q.m(:,k);
        quad = trace(W*st.S2(:,:,k)) - 2*m'*W*st.S1(:,k) + N*(m'*W*m); corr = N*D/q.kap(k);
    end
    E = E + N*(-D/2*log(2*pi) + 0.5*Elogdet(q.nu(k),W)) - 0.5*(corr + q.nu(k)*quad);
end
F = E + H - KLtheta(q,pr);
end
function v = logev(Y,pr)
[N,D] = size(Y); nu = pr.nu0 + N;
if pr.zeromean
    Wi = inv(pr.W0) + Y'*Y; kterm = 0;
else
    kap = pr.kap0 + N; xb = mean(Y)'; S = (Y-xb')'*(Y-xb'); d = xb-pr.m0;
    Wi = inv(pr.W0) + S + pr.kap0*N/kap*(d*d'); kterm = D/2*log(pr.kap0/kap);
end
v = -N*D/2*log(pi) + lgD(nu/2,D) - lgD(pr.nu0/2,D) - pr.nu0/2*logdet_(pr.W0) - nu/2*logdet_(Wi) + kterm;
end
function [Y,segs,q] = simhmm(K,D,Ts,zm)
A = 0.02*ones(K)/(K-1); A(1:K+1:end) = 0.98;
mu = 2*randn(D,K)*(1-zm); C = zeros(D,D,K); for k=1:K, C(:,:,k)=randspd(D); end
Y = []; segs = []; s0 = 0; S = [];
for T = Ts
    s = zeros(T,1); s(1) = randi(K);
    for t=2:T, s(t) = find(rand < cumsum(A(s(t-1),:)),1); end
    y = zeros(T,D); for t=1:T, y(t,:) = mu(:,s(t))' + randn(1,D)*chol(C(:,:,s(t))); end
    Y = [Y; y]; segs = [segs; s0+1 s0+T]; s0 = s0+T; S = [S; s]; %#ok<AGROW>
end
pr = defaultprior(K,D); pr.zeromean = zm; if ~zm, pr.m0 = mean(Y)'; end
st = suffstats(Y,onehot(S,K),zeros(K),zeros(1,K));
st.Xi = accumarray([S(1:end-1) S(2:end)],1,[K K]); st.g1 = ones(1,K);
q = mstep(st,pr);
end
function g = onehot(s,K), g = zeros(numel(s),K); g(sub2ind(size(g),(1:numel(s))',s(:))) = 1; end

%-------------------------------------------------------------------------- SVI helpers
function e = natparams(q,pr)
% natural-parameter-equivalent coordinates (affine in sufficient statistics)
K = numel(q.a); e.a = q.a; e.B = q.B; e.nu = q.nu;
for k=1:K
    e.kap(k) = q.kap(k); e.km(:,k) = q.kap(k)*q.m(:,k);
    e.P(:,:,k) = inv(q.W(:,:,k)) + q.kap(k)*(q.m(:,k)*q.m(:,k)');
end
end
function e = natparams_prior(pr)
K = pr.K; e.a = pr.alpha0; e.B = pr.beta0; e.nu = pr.nu0*ones(1,K); e.kap = pr.kap0*ones(1,K);
e.km = repmat(pr.kap0*pr.m0,1,K); e.P = repmat(inv(pr.W0) + pr.kap0*(pr.m0*pr.m0'),1,1,K);
end
function e = statseta(st,c)
e.a = c*st.g1; e.B = c*st.Xi; e.nu = c*st.N; e.kap = c*st.N; e.km = c*st.S1; e.P = c*st.S2;
end
function e = addeta(a,b), f = fieldnames(a); for i=1:numel(f), e.(f{i}) = a.(f{i}) + b.(f{i}); end, end
function e = blend(a,b,rho), f = fieldnames(a); for i=1:numel(f), e.(f{i}) = (1-rho)*a.(f{i}) + rho*b.(f{i}); end, end
function v = maxdiff(a,b), f = fieldnames(a); v = 0; for i=1:numel(f), v = max(v,max(abs(a.(f{i})(:)-b.(f{i})(:)))); end, end
