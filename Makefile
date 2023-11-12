.DEFAULT_GOAL := all
FOLDERS := tabr

.PHONY: format
format:
	pre-commit run --all-files

.PHONY: test
test:
	pytest

.PHONY: all
all: format
