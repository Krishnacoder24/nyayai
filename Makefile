include .env
export

test-ocr:
	uv run python tests/test_ocr.py $(FILE)
