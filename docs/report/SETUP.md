# Latex template

For current setup I use this settings with [LaTeX Workshop](https://marketplace.visualstudio.com/items?itemName=James-Yu.latex-workshop) you can copy and place them in `.vscode/settings.json`:

```json
{
  "latex-workshop.latex.recipe.default": "latexmk",
  "latex-workshop.latex.outDir": "%DIR%/tmp",
  "latex-workshop.latex.recipes": [
    {
      "name": "latexmk",
      "tools": ["latexmk"]
    },
    {
      "name": "pdflatex",
      "tools": ["pdflatex"]
    }
  ],
  "latex-workshop.latex.tools": [
    {
      "name": "latexmk",
      "command": "latexmk",
      "args": [
        "-pdf",
        "-synctex=1",
        "-interaction=nonstopmode",
        "-file-line-error",
        "-halt-on-error",
        "-outdir=%OUTDIR%",
        "-auxdir=%OUTDIR%",
        "%DOC%"
      ]
    },
    {
      "name": "pdflatex",
      "command": "pdflatex",
      "args": [
        "-synctex=1",
        "-interaction=nonstopmode",
        "-file-line-error",
        "-halt-on-error",
        "-output-directory=%OUTDIR%",
        "%DOC%"
      ]
    }
  ]
}
```
