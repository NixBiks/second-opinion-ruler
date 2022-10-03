install:
	poetry install

test:
	poetry run pytest tests --cov --cov-report=xml
	poetry run pyright

style:
	poetry run black .
	poetry run isort .

lint:
	poetry run black --check .
	poetry run isort --check-only .

clean:
	rm -rf dist