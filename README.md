# One Unified Representation — Project Page

Project page for *One Unified Representation: Resolving the Appearance-Semantics Dilemma via Structural Regularization*
(NeurIPS 2026). Plain HTML + CSS + a little JS, with no build step.

This branch (`gh-pages`) holds only the website and is served at **https://reid-yeung.github.io/SSR/**.
The code lives on the `main` branch of the same repository.

## Files

```
index.html              # All page content (search for TODO to find what is still open)
static/css/style.css    # Styles; colors are the variables at the top of :root
static/js/main.js       # Progress nav, light/dark toggle, selection highlighter, BibTeX copy
static/images/          # Paper figures
static/images/logos/    # Affiliation logos
```

## Page structure

Hero (gradient method name and subtitle, authors, affiliation logos, Paper / Code / Citation buttons, one-line lede,
three key numbers) → Abstract → Key takeaways → Method (the two paper figures, centered at 85% width) →
Results (overview table, then Reconstruction, Perception and Generation subsections) → Citation.

## Editing

1. **Title**: the method name goes in `.wordmark` (gradient), the rest in `.tsub` (grey). Use `&#8209;` for hyphens
   that should not break across lines.
2. **Authors and affiliations**: authors with a homepage use `<a href>` and show in blue;
   authors without one use `<span class="nm">`. The `<sup>` numbers refer to the affiliations below them.
3. **Lede and key numbers**: the three colored keywords in the lede (`kw-b`, `kw-a`, `kw-m`) match the three stat
   cells, the Results table group headers and the subsection markers (`t-recon`, `t-percep`, `t-gen`).
4. **Figures**: convert a PDF figure to PNG with
   `pdftoppm -png -singlefile -scale-to-x 2400 -scale-to-y -1 figure.pdf figure`.
5. **Tables**: `tr.ours` highlights our method, `sup.delta` is the red improvement label, `tr.group` is a group header
   row, `tr.split` draws a divider under its row, and `narrow` on the wrapper centers a smaller table.
6. **Progress nav**: each link in `<nav class="pnav">` points to a section `id`; update it when adding or removing
   sections.
7. **Colors**: change `--c-a` (blue), `--c-b` (rose) and `--c-m` (navy) at the top of `static/css/style.css`. They are
   taken from the paper figures; dark mode needs no separate change.

## Local preview

```bash
python3 -m http.server 8000
# open http://localhost:8000
```

## Deploying to GitHub Pages

In the repository, go to Settings → Pages, set Source to `Deploy from a branch`, and pick `gh-pages` with the
`/ (root)` folder. Every push to `gh-pages` then updates the site within a minute or two. GitHub Pages on a private
repository requires a paid plan, so make the repository public first.

## Before launch (TODO)

- [ ] Link for the Paper button in `index.html`
- [ ] BibTeX: once the paper is published, replace the whole `#bibtex-code` block with the official entry and remove
      every `TODO`
