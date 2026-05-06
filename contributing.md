# Contributing to GCT‑TTS

Thank you for considering contributing to GCT‑TTS!  
We appreciate your time and effort in helping to improve this experimental project.

## How Can I Contribute?

### Reporting Bugs
- **Check existing issues** to avoid duplicates.
- Use the [Bug Report](https://github.com/StellariaStudio/gct-tts/issues/new?template=bug_report.md) template if available, or describe:
  - Environment (OS, Python version, PyTorch version, GPU)
  - Steps to reproduce
  - Expected vs. actual behaviour
  - Any error messages or logs

### Suggesting Enhancements
- Open a **Feature Request** issue.
- Explain why this feature would be useful and, if possible, suggest a high‑level approach.

### Pull Requests
1. **Fork** the repository and create a new branch.
2. **Make your changes** – keep them focused, add comments where needed.
3. **Follow coding style** – use `black` for formatting, `flake8` for linting (we’ll add `.pre-commit-config.yaml` soon).
4. **Write / update tests** if relevant (test infrastructure is under construction).
5. **Commit** with clear messages (e.g. “fix: mel denormalization in inference”).
6. **Push** and open a pull request. Link any related issues.

### Priority Areas for Contribution
- 🔥 **Solving the “silent output” problem** – debugging mel processing, vocoder integration.
- Improving validation and evaluation pipelines.
- Adding HiFi‑GAN / BigVGAN vocoder wrapper.
- Enhancing sparse GCN performance.
- Documentation and tutorials.

## Code of Conduct
We adhere to a [Code of Conduct](./CODE_OF_CONDUCT.md) (to be added) – be respectful and inclusive.

## Questions?
Feel free to start a [Discussion](https://github.com/StellariaStudio/gct-tts/discussions) or contact the maintainers.

Thank you!  
**Stellaria Studio**