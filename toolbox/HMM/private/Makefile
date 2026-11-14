#!/usr/bin/env make -f
# HMM toolbox Makefile called by {SPM}/src/Makefile
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
