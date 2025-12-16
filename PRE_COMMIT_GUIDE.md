# Pre-commit Hooks Guide

This project uses pre-commit hooks to ensure code quality and consistency before commits. The hooks automatically format code, check for issues, and enforce best practices.

## Overview

Pre-commit hooks run automatically before every commit to:
- **Format code** with Black (Python code formatter)
- **Sort imports** with isort
- **Remove unused imports/variables** with autoflake
- **Check code quality** with flake8
- **Detect security issues** with bandit
- **Fix common issues** (trailing whitespace, end-of-file, etc.)

## Installation

### 1. Install pre-commit in your environment

```bash
# Activate your conda environment
conda activate llm_behavior_test

# pre-commit should already be installed, but if not:
pip install pre-commit
```

### 2. Install the git hooks

```bash
# This installs the hooks into .git/hooks/pre-commit
pre-commit install
```

That's it! The hooks will now run automatically before every commit.

## Usage

### Automatic Mode (Recommended)

Once installed, hooks run automatically when you commit:

```bash
git add .
git commit -m "Your commit message"
# Hooks run automatically here
```

If hooks find issues:
- **Auto-fixable issues** (formatting, imports) are fixed automatically
- **Manual fixes required** (code quality issues) will prevent the commit

After auto-fixes, you need to:
1. Review the changes: `git diff`
2. Stage the changes: `git add .`
3. Commit again: `git commit -m "Your commit message"`

### Manual Mode

Run hooks manually without committing:

```bash
# Run on all files
pre-commit run --all-files

# Run on staged files only
pre-commit run

# Run specific hook
pre-commit run black --all-files
pre-commit run flake8 --all-files
```

### Update Hooks

Update to the latest versions of all hooks:

```bash
pre-commit autoupdate
```

### Skip Hooks (Use Sparingly!)

To commit without running hooks (not recommended):

```bash
git commit --no-verify -m "Emergency fix"
```

## Hooks Configuration

### Active Hooks

1. **General File Checks** (pre-commit-hooks)
   - Check for large files (>1MB)
   - Detect merge conflicts
   - Validate YAML, JSON, TOML
   - Fix end-of-file issues
   - Remove trailing whitespace
   - Detect debug statements
   - Check for private keys

2. **autoflake**
   - Removes unused imports
   - Removes unused variables
   - Removes duplicate keys

3. **isort**
   - Sorts Python imports
   - Uses Black-compatible profile

4. **Black**
   - Python code formatter
   - Line length: 88 characters
   - PEP 8 compliant

5. **flake8**
   - Python linter
   - Checks code quality
   - Detects common errors
   - Includes flake8-docstrings and flake8-bugbear plugins

6. **bandit**
   - Security vulnerability scanner
   - Detects common security issues
   - Skips test files

### Optional Hooks (Commented Out)

These hooks are available but commented out for performance:

- **mypy**: Static type checker
- **interrogate**: Docstring coverage checker

To enable them, uncomment in `.pre-commit-config.yaml`

## Configuration Files

### `.pre-commit-config.yaml`
Main pre-commit configuration. Defines which hooks to run and their settings.

### `pyproject.toml`
Configuration for Black, isort, pytest, coverage, and mypy.

```toml
[tool.black]
line-length = 88
target-version = ['py310']

[tool.isort]
profile = "black"
line_length = 88
```

### `.flake8`
Flake8 linter configuration.

```ini
[flake8]
max-line-length = 88
extend-ignore = E203, W503, E501, D100, D101, D102, D103, D104
max-complexity = 15
```

### `.bandit`
Bandit security scanner configuration.

```yaml
exclude_dirs:
  - /tests/
skips:
  - B101  # assert statements
```

## Common Issues and Solutions

### Issue: "Files were modified by this hook"

**Cause**: Hooks auto-fixed your code (formatting, imports, etc.)

**Solution**:
```bash
# Review changes
git diff

# Stage the changes
git add .

# Commit again
git commit -m "Your message"
```

### Issue: Flake8 errors prevent commit

**Cause**: Code quality issues that can't be auto-fixed

**Solution**: Fix the issues manually or adjust `.flake8` configuration

```bash
# See specific errors
pre-commit run flake8 --all-files

# Fix the issues in your code
# Then commit again
```

### Issue: Hook installation fails

**Cause**: Missing dependencies or wrong Python version

**Solution**:
```bash
# Ensure you're in the right environment
conda activate llm_behavior_test

# Reinstall pre-commit
pip install --upgrade pre-commit

# Clear cache and reinstall hooks
pre-commit clean
pre-commit install
```

### Issue: Hooks take too long

**Cause**: Running on all files

**Solution**: Hooks only run on staged files by default. For large commits:
```bash
# Stage files incrementally
git add specific_file.py
git commit -m "Update specific file"
```

## Workflow Best Practices

### Recommended Git Workflow

```bash
# 1. Make changes to your code
vim your_file.py

# 2. Stage your changes
git add your_file.py

# 3. Commit (hooks run automatically)
git commit -m "Add new feature"

# 4. If hooks auto-fix:
git add .
git commit -m "Add new feature"

# 5. Push
git push
```

### Before Committing Large Changes

Run hooks manually first to see what needs fixing:

```bash
# Run on all files
pre-commit run --all-files

# Fix issues
# Stage and commit
```

### CI/CD Integration

Add to your CI pipeline (`.github/workflows/lint.yml`):

```yaml
- name: Run pre-commit
  run: |
    pip install pre-commit
    pre-commit run --all-files
```

## Customization

### Disable Specific Hook

Edit `.pre-commit-config.yaml` and comment out the hook:

```yaml
# - repo: https://github.com/PyCQA/bandit
#   rev: 1.7.6
#   hooks:
#     - id: bandit
```

### Add New Hook

Add to `.pre-commit-config.yaml`:

```yaml
  - repo: https://github.com/new-hook/repo
    rev: v1.0.0
    hooks:
      - id: hook-name
```

### Modify Hook Arguments

Edit the `args` section in `.pre-commit-config.yaml`:

```yaml
  - repo: https://github.com/psf/black
    rev: 24.1.1
    hooks:
      - id: black
        args: ['--line-length=100']  # Change line length
```

## Troubleshooting

### Clear Pre-commit Cache

```bash
pre-commit clean
pre-commit install
```

### Uninstall Hooks

```bash
pre-commit uninstall
```

### Debug Hook Execution

```bash
# Verbose mode
pre-commit run --all-files --verbose

# Run specific hook with verbose
pre-commit run black --all-files --verbose
```

### Check Hook Versions

```bash
pre-commit --version
```

## IDE Integration

### VS Code

Install the "Pre-commit" extension and add to `settings.json`:

```json
{
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "editor.formatOnSave": true
}
```

### PyCharm

1. Go to Settings → Tools → External Tools
2. Add Black, isort, and flake8 as external tools
3. Configure file watchers to run on save

## Resources

- [Pre-commit Documentation](https://pre-commit.com/)
- [Black Documentation](https://black.readthedocs.io/)
- [isort Documentation](https://pycqa.github.io/isort/)
- [Flake8 Documentation](https://flake8.pycqa.org/)
- [Bandit Documentation](https://bandit.readthedocs.io/)

## Summary

Pre-commit hooks help maintain code quality automatically. They:
- ✅ Save time by auto-fixing common issues
- ✅ Enforce consistent code style across the team
- ✅ Catch bugs and security issues early
- ✅ Make code reviews faster and more focused
- ✅ Reduce CI/CD failures

Once installed, they work seamlessly in your development workflow!
