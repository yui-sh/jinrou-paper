# The Inverted Tell: How AI Deception Reverses Human Linguistic Patterns in Werewolf Games

**Mingxuan He** (University of Chicago & Gauntlet) and **Yui** (Independent)

## Abstract

We analyze 882,510 chat messages from 31,479 games of AI Werewolf (Jinrou) played by eight frontier large language models on the Kaggle Arena — the largest corpus of AI-vs-AI natural language deception to date. We find that AI deception exhibits *systematically inverted* linguistic patterns compared to well-established human deception findings: where human liars use fewer first-person pronouns, AI liars use more; where human liars produce shorter statements, AI liars produce longer ones; where human liars show markers of reduced cognitive complexity, AI liars show increased epistemic hedging.

We further show that the most successful AI deceivers are those whose behavioral shifts are *smallest* — the best AI liars are the ones who change least.

## Structure

- `main.tex` — Main document
- `sections/` — Paper sections
- `figures/` — Publication-quality figures (PDF)
- `generate_figures.py` — Figure generation script
- `references.bib` — Bibliography
- `neurips.sty` — NeurIPS style file

## Building

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## License

CC BY 4.0
