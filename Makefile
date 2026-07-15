all: build/thesis.pdf


TeXOptions = -lualatex \
			 -interaction=nonstopmode \
			 -halt-on-error \
			 -output-directory=build

FIGURES = content/bilder/aufbau.pdf

build/thesis.pdf: FORCE $(FIGURES) | build
	latexmk $(TeXOptions) thesis.tex

content/bilder/%.pdf: content/bilder/%.tex
	latexmk -lualatex -interaction=nonstopmode -halt-on-error -output-directory=content/bilder $<

FORCE:

build:
	mkdir -p build/

clean:
	rm -rf build
	latexmk -C -output-directory=content/bilder content/bilder/aufbau.tex
