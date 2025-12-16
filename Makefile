PY_SOURCE_FILES=understanding/ llm_behavior_adaptation/ scripts/ #this can be modified to include more files

install: package
	pip install -e .[dev,data_process,training]

test:
	pytest tests -vv -s

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache
	find . -name '*.pyc' -type f -exec rm -rf {} +
	find . -name '__pycache__' -exec rm -rf {} +

package: clean
	python setup.py sdist bdist_wheel

format:
	autoflake --in-place --remove-all-unused-imports --remove-unused-variables --remove-duplicate-keys --expand-star-imports --recursive ${PY_SOURCE_FILES}
	isort ${PY_SOURCE_FILES}
	black --line-length 119 ${PY_SOURCE_FILES}

lint:
	isort --check --diff ${PY_SOURCE_FILES}
	black --check --diff --line-length 119 ${PY_SOURCE_FILES}
	flake8 --config .flake8 ${PY_SOURCE_FILES}

# Pre-commit hooks
pre-commit-install:
	pre-commit install

pre-commit-uninstall:
	pre-commit uninstall

pre-commit-run:
	pre-commit run --all-files

pre-commit-update:
	pre-commit autoupdate

pre-commit-clean:
	pre-commit clean

# Run all code quality checks
check: lint pre-commit-run test
