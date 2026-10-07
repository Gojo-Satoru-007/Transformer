# Define variables to allow overriding (e.g., make train PYTHON=python3)
PYTHON = python
PIP = pip
PYTEST = pytest

# Mark targets that do not represent physical files
.PHONY: install test prepare train-tokenizer train generate all clean

# Install dependencies
install:
	$(PIP) install -r requirements.txt

# Run the test suite
test:
	$(PYTEST) tests/

# Prepare the dataset
prepare:
	$(PYTHON) prepare_data.py

# Train the custom BPE tokenizer
train-tokenizer:
	$(PYTHON) train_tokenizer.py

# Train the transformer model
train:
	$(PYTHON) train.py

# Generate text using the trained model
generate:
	$(PYTHON) generate.py

# Run the complete data pipeline sequentially
all: prepare train-tokenizer train generate

# Clean Python caches and compiled files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache