.PHONY: install data train pipeline test lint dashboard clean

install:
	python -m pip install -r requirements-dev.txt

data:
	python run.py data

train:
	python run.py train

pipeline:
	python run.py all

test:
	python -m pytest -q

lint:
	python -m ruff check src tests dashboard run.py

dashboard:
	python run.py dashboard

clean:
	rm -rf .pytest_cache .ruff_cache reports/figures/*.png
