# Simple Makefile for LaTeX compilation
MAIN = main
BIB = references

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex $(BIB).bib
	pdflatex $(MAIN).tex || true
	bibtex $(MAIN) || true
	pdflatex $(MAIN).tex || true
	pdflatex $(MAIN).tex

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.fdb_latexmk *.fls *.synctex.gz

cleanall: clean
	rm -f $(MAIN).pdf