-include .env

export PYTHONPATH := $(CURDIR)

.PHONY: \
	test-ocr \
	cleanup-outputs \
	download-models \
	ingest-corpus \
	generate-data \
	train \
	evaluate

test-ocr:
	uv run python scripts/smoke_test_ocr.py $(FILE)

cleanup-outputs:
	uv run python scripts/cleanup_outputs.py $(ARGS)

download-models:
	uv run python scripts/download_models.py $(ARGS)

ingest-corpus:
	uv run python scripts/ingest_corpus.py --all

generate-data:
	uv run python scripts/generate_data.py --corpus corpus/sources/ --out data/training $(ARGS)

train:
	uv run python -m train.train

evaluate:
	uv run python -m train.evaluate