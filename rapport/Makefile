# Makefile for building the rapport (XeLaTeX + biber)
TEX = xelatex
BIB = biber
MAIN = main

all: build

build:
	$(TEX) $(MAIN).tex
	$(BIB) $(MAIN)
	$(TEX) $(MAIN).tex
	$(TEX) $(MAIN).tex

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out $(MAIN).toc $(MAIN).lof $(MAIN).lot

.PHONY: all build clean
