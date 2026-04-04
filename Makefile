.PHONY: dev-backend dev-frontend dev test-collector test-backend test example

dev-backend:
	cd backend && uv run uvicorn gradienthound_server.main:app --reload --port 8642

dev-frontend:
	cd frontend && pnpm dev

dev:
	$(MAKE) dev-backend & $(MAKE) dev-frontend

test-collector:
	cd collector && uv run pytest

test-backend:
	cd backend && uv run pytest

test:
	$(MAKE) test-collector && $(MAKE) test-backend

example:
	cd collector && uv run python ../examples/simple_cnn.py
