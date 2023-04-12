test:
	pytest

install:
	python -m pip install -e .

install-test:
	python -m pip install -e ".[test,spacy]"
	python -m pip install -e "."
	python -m spacy download en_core_web_sm

pypi:
	python setup.py sdist
	python setup.py bdist_wheel --universal
	twine upload dist/*

clean:
	rm -rf **/.ipynb_checkpoints **/.pytest_cache **/__pycache__ **/**/__pycache__ .ipynb_checkpoints .pytest_cache

check: test clean
