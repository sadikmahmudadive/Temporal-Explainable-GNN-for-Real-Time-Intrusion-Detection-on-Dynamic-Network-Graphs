# IEEE Conference Paper (LaTeX)

Main manuscript: `paper/ieee_conference_paper.tex`  
Bibliography: `paper/references.bib`

## Compile

From the repository root (PowerShell):

```powershell
cd paper
pdflatex ieee_conference_paper.tex
bibtex ieee_conference_paper
pdflatex ieee_conference_paper.tex
pdflatex ieee_conference_paper.tex
```

Notes:
- Figures are referenced from `../evaluation/plots/` via `\graphicspath`, so compile from the `paper/` directory.
- If `pdflatex`/`bibtex` are not found, install a LaTeX distribution (e.g., MiKTeX or TeX Live) and ensure it is on `PATH`.
- The manuscript is written against the currently checked-in repository artifacts. If you rerun `python -m training.run_training --config config.yml`, review the generated metrics/artifacts and refresh the tables before submission.
