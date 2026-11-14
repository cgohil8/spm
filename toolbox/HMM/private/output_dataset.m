function Dout = output_dataset(D,prefix,nchan)
% New float32 dataset named <prefix><name of D>, with nchan channels whose
% channel information is reset (labels, types and units are set by the
% caller); trials, events and other header information are copied.
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


fd   = spm_file(D.fname,'ext','.dat','prefix',prefix);
Dout = clone(D,fd,[nchan D.nsamples D.ntrials],1,1);
