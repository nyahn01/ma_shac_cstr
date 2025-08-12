# Academic Project Structure Guide

This repository follows academic best practices for Master's thesis research. Each directory serves a specific purpose in the research workflow.

## Quick Navigation

- **📖 [thesis/](thesis/)** - Thesis manuscript and writing
- **🎯 [presentations/](presentations/)** - Slides and presentation materials  
- **🖥️ [source/](source/)** - Implementation code
- **🧪 [experiments/](experiments/)** - Experimental work and configs
- **📊 [data/](data/)** - Datasets and data management
- **📈 [results/](results/)** - Generated results and analysis
- **📚 [docs/](docs/)** - Documentation and meeting notes
- **🔧 [utils/](utils/)** - Utility functions and scripts

## Research Workflow

### 1. Planning Phase
```bash
# Document research plan
docs/timeline/research_plan.md
docs/meeting_notes/YYYY-MM-DD_kickoff.md

# Set up initial experiments
experiments/exp001_baseline/
```

### 2. Implementation Phase  
```bash
# Develop core algorithms
source/agent.py
source/environment.py

# Create configurations
experiments/exp001_baseline/config.py

# Write tests
tests/test_*.py
```

### 3. Experimentation Phase
```bash
# Run experiments
python experiments/exp001_baseline/run_experiment.py

# Analyze results
results/analysis_notebooks/exp001_analysis.ipynb

# Generate figures
results/figures/thesis_figures/
```

### 4. Writing Phase
```bash
# Write thesis chapters
thesis/chapters/01_introduction.tex

# Create presentations
presentations/progress_reports/

# Generate final figures
python scripts/generate_thesis_figures.py
```

### 5. Defense Phase
```bash
# Prepare defense materials
presentations/defense/

# Final thesis compilation
thesis/compiled/final_thesis.pdf
```

## Git Workflow for Academics

### Branch Strategy
```bash
main                    # Stable, working version
chapter/introduction    # Writing branches
experiment/exp001      # Experiment branches
presentation/defense   # Presentation branches
analysis/statistics    # Analysis branches
```

### Typical Commands
```bash
# Start new work
git checkout main
git pull origin main
git checkout -b experiment/exp002-improved-gae

# Save progress
git add .
git commit -m "experiment: implement improved GAE (exp002)

Configuration:
- Learning rate: 1e-3
- GAE lambda: 0.95
- Episode length: 32

Results:
- Training converged in 15k episodes
- 12% improvement over baseline"

# Integrate successful work
git checkout main
git merge experiment/exp002-improved-gae
git tag exp002-improved-gae
git push origin main --tags
```

## Academic Commit Conventions

### Code Development
- `feat:` New features or algorithms
- `fix:` Bug fixes
- `refactor:` Code restructuring
- `test:` Adding tests

### Experiments
- `experiment:` New experimental results
- `analysis:` Data analysis and insights
- `config:` Configuration changes

### Writing
- `docs(thesis):` Thesis writing
- `docs(chapter1):` Specific chapter work
- `present:` Presentation materials

### Data and Results
- `data:` Dataset updates
- `results:` New experimental results
- `figures:` Generated visualizations

## Collaboration Guidelines

### With Supervisors
1. **Regular Updates**: Push changes frequently
2. **Clear Commits**: Descriptive commit messages
3. **PDF Access**: Include compiled documents
4. **Meeting Notes**: Document all discussions

### With Peers
1. **Code Reviews**: Use pull requests for major changes
2. **Documentation**: Comprehensive README files
3. **Reproducibility**: Clear setup instructions
4. **Sharing**: Easy experiment replication

## File Naming Conventions

### Experiments
- `exp001_baseline_shac`
- `exp002_improved_gae`
- `exp003_hyperparameter_sweep`

### Presentations
- `YYYY-MM-DD_event_description.extension`
- `2024-03-15_defense_final.pptx`
- `2024-02-28_progress_report.pdf`

### Results
- `exp001_training_metrics.json`
- `baseline_vs_improved_comparison.csv`
- `statistical_significance_tests.json`

### Figures
- `fig01_system_architecture.pdf`
- `fig02_training_curves.svg`
- `table01_results_summary.tex`

## Quality Assurance

### Code Quality
- Unit tests for all major components
- Code documentation and type hints
- Consistent formatting (black, flake8)
- Performance profiling for key algorithms

### Reproducibility
- Fixed random seeds
- Environment specifications
- Data provenance tracking
- Step-by-step reproduction guides

### Documentation
- Algorithm descriptions
- Experimental methodologies
- Results interpretation
- Future work suggestions

## Timeline Management

### Milestones
- [ ] Literature review complete
- [ ] Baseline implementation
- [ ] Core experiments finished
- [ ] Thesis first draft
- [ ] Defense preparation
- [ ] Final submission

### Regular Tasks
- Weekly supervisor meetings
- Monthly progress reports
- Quarterly committee updates
- Continuous code development
- Regular result analysis

This structure supports the entire academic research lifecycle from initial planning through final defense and publication.
