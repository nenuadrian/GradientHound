.PHONY: install test lint example clean

install:
	uv pip install -e ".[dev,torch]"

test:
	uv run pytest tests/

lint:
	uv run ruff check src/

example:
	uv run python examples/simple_cnn.py

clean:
	rm -rf build/ dist/ *.egg-info src/*.egg-info
