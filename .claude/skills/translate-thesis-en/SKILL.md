---
name: translate-thesis-en
description: Sync the English thesis translation in content/en/ with recent edits to the German content/*.tex files, then rebuild build/thesis_en.pdf.
---

# Translate thesis to English

This project keeps two parallel versions of the bachelor thesis:

- German source: `thesis.tex` + `content/*.tex`
- English translation: `thesis_en.tex` + `content/en/*.tex` (same filenames, mirrored 1:1)

This skill brings the English translation back in sync after the German source
has changed, and rebuilds the English PDF.

## Steps

1. **Find what changed.** Run `git status` and `git diff -- content/*.tex thesis.tex`
   (and `git log` if needed) to see which German content files were added,
   removed, or edited since the English translation was last updated. Compare
   file lists in `content/` (excluding `bilder/`, `hints.tex`, and non-`.tex`
   files) against `content/en/` — a German file with no English counterpart is
   new and needs a fresh translation file.

2. **Translate only the diff.** For each changed German file, open it side by
   side with its `content/en/<same-name>.tex` counterpart and update the
   English text to match the new meaning. Do not re-translate untouched
   paragraphs — preserve existing English wording where the German didn't
   change, to avoid needless churn in the English document.

3. **Preserve LaTeX structure exactly.** Never translate or alter:
   - Commands, environments, and their arguments in `{...}`/`[...]` (e.g.
     `\includegraphics`, `\autoref`, `\SIrange`, `\si{...}`, `\label{...}`).
   - `\label{...}` keys — keep them byte-identical between the German and
     English files so both documents can be cross-referenced consistently.
   - `\cite{...}` keys and citation content.
   - Math mode content.
   - File paths (e.g. `content/bilder/...`).

   Only the natural-language prose (chapter/section titles, running text,
   captions) gets translated.

4. **New German files.** If a new file appears under `content/` (e.g. a new
   numbered chapter), create the matching file under `content/en/` with a
   full translation, and add the corresponding `\input{content/en/...}` line
   to `thesis_en.tex` in the same position as in `thesis.tex`.

5. **Deleted German files.** If a German content file was removed, remove its
   English counterpart and the matching `\input` line in `thesis_en.tex`.

6. **Rebuild and check.** Run `make en` and confirm `build/thesis_en.pdf` is
   produced without LaTeX errors. If latexmk reports errors, read
   `build/thesis_en.log` to fix them (usually a missing/renamed label or a
   stray unescaped character introduced during translation).

7. **Report back.** Summarize which English files were updated/created/
   removed, in one or two sentences — don't dump the full translated text
   unless asked.

## When NOT to use this skill

If the user asks to translate the whole thesis from scratch (not a sync of an
existing translation), just do the translation directly instead — this skill
is specifically for keeping an existing English version current after edits
to the German original.
