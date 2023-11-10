.DEFAULT_GOAL := all
FOLDERS := tabr

.PHONY: lint
lint:
	ruff check $(FOLDERS)
	mypy $(FOLDERS)

.PHONY: format
format: lint
	ruff fix $(FOLDERS)


.PHONY: all
all: format
