include .env
export

test-ocr:
	uv run python tests/test_ocr.py $(FILE)

cleanup-outputs:
	uv run python scripts/cleanup_outputs.py $(ARGS)