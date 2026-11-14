#!/usr/bin/env make -f
# Builds the HMM toolbox MEX file (not part of the SPM build in src/Makefile)
# Run 'make' here with mex on the path, or spm_hmm_compile in MATLAB.
#
# Chetan Gohil
# Copyright (C) 2026 NDCN, University of Oxford

include ../../../src/Makefile.var

SPMMEX = forward_backward_mex.$(MEXEXT)

all: $(SPMMEX)
	@:

clean:
	@:
ifeq (mex,$(MEXEXT))
	$(DEL) $(subst .$(MEXEXT),.o,$(SPMMEX))
endif

distclean: clean
	$(DEL) $(SPMMEX)

install:
	@:

tarball: all
	$(TAR) cf spm_mex.tar $(SPMMEX)

%.$(MEXEXT) : %.c
	$(MEX) $< $(MEXEND)
