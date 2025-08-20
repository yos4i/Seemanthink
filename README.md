# Semanthink

An automated Semantle word guessing game solver using machine learning techniques and clustering algorithms.

## Overview

Semanthink is a Python-based tool that automatically solves Semantle puzzles by leveraging Word2Vec embeddings and intelligent clustering strategies. The system uses semantic similarity to make strategic guesses and converge on the target word.

## Features

- Automated Semantle puzzle solving
- Smart and random medoid clustering strategies
- Word2Vec integration for semantic analysis
- Cross-platform support (Windows/Linux)
- Configurable clustering parameters

## Quick Start

1. Install dependencies: `pip install -r requirements.txt`
2. Run the solver: `python components/AutomatedSemantleSolver.py`

For detailed setup and usage instructions, see [INSTRUCTIONS.md](INSTRUCTIONS.md).

## Requirements

- Python 3.12+
- Word2Vec model file (GoogleNews-vectors-negative300.bin)
- English vocabulary file (English-Words_Semantle_filtered.txt)

## Components

The project includes various Python modules in the `components/` directory for different aspects of the Semantle solving process.

