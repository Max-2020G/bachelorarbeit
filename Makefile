all: build/thesis.pdf

en: build/thesis_en.pdf

TeXOptions = -lualatex \
			 -interaction=nonstopmode \
			 -halt-on-error \
			 -output-directory=build

FIGURES = content/bilder/aufbau.pdf

PLOT_SCRIPTS = $(wildcard content/plots/*.py)
PLOT_STAMPS = $(PLOT_SCRIPTS:.py=.stamp)

build/thesis.pdf: FORCE $(FIGURES) $(PLOT_STAMPS) | build
	latexmk $(TeXOptions) thesis.tex

build/thesis_en.pdf: FORCE $(FIGURES) $(PLOT_STAMPS) | build
	latexmk $(TeXOptions) thesis_en.tex

content/bilder/%.pdf: content/bilder/%.tex
	latexmk -lualatex -interaction=nonstopmode -halt-on-error -output-directory=content/bilder $<

# reruns a plot script only if it changed since its last run
content/plots/%.stamp: content/plots/%.py
	cd content/plots && python3 $(notdir $<)
	touch $@

FORCE:

build:
	mkdir -p build/

clean:
	rm -rf build
	rm -f content/plots/*.stamp
	latexmk -C -output-directory=content/bilder content/bilder/aufbau.tex
