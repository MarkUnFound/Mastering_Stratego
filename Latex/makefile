all : clean sp

sp: sp.tex
	pdflatex sp.tex
	bibtex sp.aux
	pdflatex sp.tex
	pdflatex sp.tex

clean:
	find . -type f -name '*.aux' -delete
	rm -rf *.bbl *.blg *.idx *.lof *.log *.toc *.synctex.gz *.lot
