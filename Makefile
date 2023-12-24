# Makefile

.PHONY: setup clean run

setup:
	@echo "Setting up Python virtual environment..."
	python3 -m venv venv
	@echo "Activating virtual environment and installing dependencies..."
	@. venv/bin/activate; \
	pip install --upgrade pip; \
	pip install -r requirements.txt; \
	python -m nltk.downloader punkt; \
	python -m nltk.downloader stopwords; \
	python -m spacy download en_core_web_sm; \
	echo "Setup complete."

run:
	@echo "Running ChatThreadAnalyzer.py in virtual environment..."
	@. venv/bin/activate; \
	python ChatThreadAnalyzer.py processed.json analyzed.json

clean:
	@echo "Cleaning up..."
	rm -rf venv
	@echo "Cleaned up the virtual environment."
