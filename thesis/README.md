# Thesis Writing

This directory contains the Master's thesis manuscript and related writing materials.

## Structure

### LaTeX Setup (Recommended)
```
thesis/
├── main.tex              # Main document
├── chapters/             # Individual chapters
├── figures/              # Thesis figures
├── tables/               # Thesis tables
├── bibliography.bib      # References
├── appendix/             # Appendices
└── compiled/             # Generated PDFs
```

### Chapter Organization
- `01_introduction.tex` - Problem statement, objectives, contributions
- `02_literature_review.tex` - Related work and background
- `03_methodology.tex` - Theoretical framework and approach
- `04_implementation.tex` - System design and implementation details
- `05_experiments.tex` - Experimental setup and methodology
- `06_results.tex` - Results and analysis
- `07_conclusion.tex` - Conclusions and future work

## Writing Workflow

### Git Workflow for Writing
```bash
# Start new chapter
git checkout main
git checkout -b chapter/literature-review
# ... write chapter ...
git add thesis/chapters/02_literature_review.tex
git commit -m "docs(thesis): complete literature review draft"

# Regular progress commits
git add thesis/chapters/03_methodology.tex
git commit -m "docs(chapter3): add problem formulation section"

# Merge completed chapters
git checkout main
git merge chapter/literature-review
```

### Best Practices

1. **One Chapter per Branch**: Work on chapters independently
2. **Frequent Commits**: Commit logical sections, not just at end of day
3. **Descriptive Messages**: Use clear commit messages for writing progress
4. **Backup Compiled PDFs**: Include generated PDFs for supervisor access
5. **Reference Management**: Use BibTeX for consistent citations

### Collaboration with Supervisors

1. **Share PDFs**: Include compiled PDFs in version control
2. **Clear Milestones**: Tag major draft versions
3. **Track Comments**: Use Git to track addressing feedback
4. **Regular Updates**: Push changes frequently for supervisor access

### File Naming
- Use descriptive, lowercase names with underscores
- Include version numbers for major drafts
- Date stamp important versions

## Templates

Include standard university templates for:
- Title page format
- Abstract format
- Declaration page
- Table of contents style
- Bibliography style
