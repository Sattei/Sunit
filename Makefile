.PHONY: dev-up dev-down health test smoke clean-runtime

dev-up:
	./scripts/dev_up.sh

dev-down:
	./scripts/dev_down.sh

health:
	./scripts/health_check.sh

test:
	PYTHONPATH=app app/.venv/bin/python -m pytest app/tests -q
	npm run build --prefix app/frontend

smoke:
	./scripts/smoke_test.sh

clean-runtime:
	./scripts/dev_down.sh
	find .runtime -mindepth 1 -maxdepth 1 ! -name logs -exec rm -rf -- {} +
