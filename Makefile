# Define variables to allow overriding (e.g., make train PYTHON=python3)
PYTHON = python
PIP = pip
PYTEST = pytest

# Mark targets that do not represent physical files
.PHONY: install test prepare train-tokenizer train train-nplm compare generate all clean

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

# Train the NPLM baseline (Bengio et al., 2003) -- parameter-matched to the transformer
# (2,840,861 vs. 2,840,480 params) and run at batch_size=4096 so both models see the same
# number of next-token predictions per step (transformer: 32 seqs x 128 positions = 4,096).
train-nplm:
	$(PYTHON) train_nplm.py --batch_size 4096

# Train both models back-to-back with matching step budgets, for a direct comparison.
# Override e.g. `make compare STEPS=3000` to change the step count for both runs.
STEPS ?= 3000
compare:
	$(PYTHON) train.py --max_steps $(STEPS) --out checkpoints/transformer.pt
	$(PYTHON) train_nplm.py --max_steps $(STEPS) --batch_size 4096 --out checkpoints/nplm.pt

# Generate text using the trained transformer model
generate:
	$(PYTHON) generate.py

# Run the complete data pipeline sequentially, training both the transformer and the
# NPLM baseline, then generating from the transformer checkpoint.
all: prepare train-tokenizer train train-nplm generate

# Clean Python caches, compiled files, checkpoints, and training logs
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf checkpoints
	rm -rf logs