test:
	pytest

coverage:
	pytest --cov

install:
	python -m pip install -e .

install-test:
	python -m pip install -e ".[dev]"

docs:
	mkdocs serve

pypi:
	python setup.py sdist
	python setup.py bdist_wheel --universal
	twine upload dist/*

clean:
	rm -rf **/.ipynb_checkpoints **/.pytest_cache **/__pycache__ **/**/__pycache__ .ipynb_checkpoints .pytest_cache

check: test clean
