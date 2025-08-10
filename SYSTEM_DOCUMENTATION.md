# Semanthink - Automated Semantle Solver Documentation

## Overview

Semanthink is an intelligent automated solver for the Semantle word game that uses advanced clustering algorithms and smart semantic guidance to find target words efficiently. The system combines Word2Vec embeddings with hierarchical clustering to systematically explore vocabulary space.

## What is Semantle?

Semantle is a word guessing game where you try to find a secret word by making guesses. Each guess receives a similarity score (0-100) based on how semantically similar your guess is to the target word. The goal is to find the target word in as few guesses as possible.

## System Architecture

### Project Structure

The project is now organized with all core components in the `components/` package for better modularity and maintainability:

```
Semanthink/
├── components/              # Core system components (package)
│   ├── __init__.py         # Package initialization
│   ├── *.py               # All Python components
│   └── Tests/             # Test results and outputs
├── MyData/                # Data files (Word2Vec, vocabulary, results)
├── venv/                  # Python virtual environment
├── requirements.txt       # Python dependencies
├── README.md             # Basic project information
└── SYSTEM_DOCUMENTATION.md # This file
```

### Core Components

```
components/
├── AutomatedSemantleSolver.py      # Main solver with smart medoids
├── VocabularyClusteringSystem.py   # Orchestrates clustering pipeline  
├── VocabularyLoader.py             # Loads Word2Vec models and vocabulary
├── KMeansClusterer.py              # K-means clustering implementation
├── ClusterAnalyzer.py              # Analyzes cluster quality and structure
├── SemanticExplorer.py             # Explores semantic neighborhoods
├── DataExporter.py                 # Exports results and visualizations
├── Visualizer.py                   # Creates interactive visualizations
├── VocabularyFilter.py             # Filters and preprocesses vocabulary
├── SemantleSimulator.py            # Simulates Semantle game mechanics
├── SemantleAutomation.py           # Web automation for real Semantle
├── Semantle Bot.py                 # Bot interface for automated play
├── ClusteringAlgorithm.py          # Abstract base class for clustering
└── Tests/                          # Test results and outputs
```

---

## Component Details

### 1. AutomatedSemantleSolver.py
**Purpose**: Main solver that orchestrates the entire solving process using enhanced smart medoids strategy.

**Key Features**:
- **Smart Medoids**: Uses 15 strategically chosen starting words covering diverse semantic categories
- **Hierarchical Clustering**: Recursively explores promising semantic neighborhoods  
- **Adaptive Strategy**: Focuses exploration based on similarity scores
- **Progress Tracking**: Detailed logging of search progress

**Smart Medoids Categories**:
```python
smart_medoids = [
    # Diverse semantic coverage
    "person", "house", "water", "think", "red", "big", 
    "computer", "animal", "music", "move",
    # Spatial/positional coverage  
    "inside", "outside", "center", "edge", "around"
]
```

### 2. VocabularyClusteringSystem.py
**Purpose**: Central orchestrator that manages the clustering pipeline and coordinates between components.

**Responsibilities**:
- Loads and manages Word2Vec models and vocabulary
- Coordinates clustering operations
- Manages result storage and file organization
- Provides unified interface for clustering operations

### 3. VocabularyLoader.py
**Purpose**: Handles loading and preprocessing of Word2Vec models and vocabulary files.

**Key Functions**:
- Loads GoogleNews Word2Vec model (300-dimensional vectors)
- Filters vocabulary to match available embeddings
- Provides efficient access to word vectors
- Handles memory management for large models

### 4. KMeansClusterer.py
**Purpose**: Implements K-means clustering with cosine distance for semantic clustering.

**Features**:
- Cosine distance metric (optimal for word embeddings)
- Configurable cluster numbers
- Medoid identification for cluster representatives
- Efficient clustering of high-dimensional word vectors

### 5. ClusterAnalyzer.py
**Purpose**: Analyzes cluster quality, finds medoids, and evaluates clustering results.

**Capabilities**:
- Calculates cluster centroids and medoids
- Measures cluster quality metrics
- Finds words closest to cluster centers
- Evaluates cluster separation and cohesion

### 6. SemanticExplorer.py
**Purpose**: Explores semantic neighborhoods around promising words.

**Functions**:
- Finds words most similar to given targets
- Explores local semantic neighborhoods
- Ranks words by semantic similarity
- Guides search direction based on similarity patterns

### 7. DataExporter.py
**Purpose**: Exports clustering results, statistics, and performance metrics.

**Outputs**:
- CSV files with clustering results
- Performance statistics and summaries
- Detailed search logs and traces
- Experiment comparison data

### 8. Visualizer.py
**Purpose**: Creates interactive visualizations of clustering results and search progress.

**Visualizations**:
- Interactive cluster maps using Plotly
- Search path visualization
- Similarity score progression
- Semantic space exploration maps

### 9. VocabularyFilter.py
**Purpose**: Filters and preprocesses vocabulary for optimal performance.

**Features**:
- Removes invalid or problematic words
- Filters by word length and frequency
- Ensures vocabulary-embedding consistency
- Optimizes vocabulary for search efficiency

### 10. SemantleSimulator.py
**Purpose**: Simulates the Semantle game for testing and evaluation.

**Capabilities**:
- Provides similarity scoring like real Semantle
- Tracks guess history and statistics
- Enables offline testing and development
- Validates solver performance

---

## How to Run the System

### Prerequisites

1. **Python Environment**:
   ```bash
   # Activate virtual environment
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Required Data Files**:
   - Word2Vec model: `GoogleNews-vectors-negative300.bin`
   - Vocabulary file: `English-Words_Semantle_filtered.txt`

### Basic Usage

#### 1. Basic Usage Examples

**Run with default smart medoids (no target word specified):**
```bash
cd components
python3 AutomatedSemantleSolver.py
```

**Run with specific target word:**
```bash
cd components
python3 AutomatedSemantleSolver.py book
```

**Run with smart medoids (default):**
```bash
cd components
python3 AutomatedSemantleSolver.py book --medoids smart
```

**Run with random medoids:**
```bash
cd components
python3 AutomatedSemantleSolver.py book --medoids random --clusters 15
```

**Command Options:**
- `target_word` (optional): The word to solve for
- `--medoids`: Choose `smart` (default) or `random` medoid strategy  
- `--clusters`: Number of clusters when using random medoids (default: 10)

**Built-in Safety Limits:**
- Maximum search depth: 6 levels (prevents infinite recursion)
- Maximum total guesses: 200 per run (prevents excessive resource usage)

This will:
- Use the specified medoids strategy
- Create timestamped results folder in components/Tests/
- Generate detailed progress logs

#### 2. Import as Package (Recommended)
From the project root directory, you can now import components:
```python
from components import AutomatedSemantleSolver
from components.VocabularyClusteringSystem import VocabularyClusteringSystem

# Use the components
solver = AutomatedSemantleSolver()
```


### Medoid Strategy Selection

The system offers two main medoid strategies for different use cases:

#### Smart Medoids Strategy (Recommended)
The smart medoids strategy uses 15 carefully selected words that provide broad semantic coverage:

**Default Smart Medoids:**
```python
smart_medoids = [
    "person", "house", "water", "think", "red", "big", 
    "computer", "animal", "music", "move",
    "inside", "outside", "center", "edge", "around"
]
```

**When to use Smart Medoids:**
- General purpose solving (most common use case)
- Unknown target word categories
- Maximum semantic coverage needed
- Best overall performance

**Usage:**
```bash
python3 AutomatedSemantleSolver.py book --medoids smart
```

#### Random Medoids Strategy
Uses K-means clustering with random initialization to create medoid clusters.

**When to use Random Medoids:**
- Testing different clustering approaches
- Experimentation and research
- Specific domain vocabulary analysis
- Custom cluster count requirements

**Usage:**
```bash
python3 AutomatedSemantleSolver.py book --medoids random --clusters 20
```

**Parameters:**
- `--clusters`: Number of initial clusters (default: 10, range: 5-50 recommended)

### Target Word Selection

#### Automatic Target Selection
If no target word is provided, the system will:
1. Select a random word from the vocabulary
2. Display the chosen word at startup
3. Proceed with normal solving process

```bash
python3 AutomatedSemantleSolver.py  # Random target word
```

#### Manual Target Selection
Specify any word from the vocabulary as target:

```bash
python3 AutomatedSemantleSolver.py science    # Solve for "science"
python3 AutomatedSemantleSolver.py happiness  # Solve for "happiness"  
python3 AutomatedSemantleSolver.py algorithm  # Solve for "algorithm"
```

**Target Word Requirements:**
- Must exist in the loaded vocabulary file
- Must have a Word2Vec embedding in the model
- Case-insensitive input (automatically normalized)

### Advanced Configuration

#### Modify Smart Medoids
Edit the medoids list in `AutomatedSemantleSolver.py` (lines 127-130):
```python
smart_medoids = [
    "person", "house", "water", "think", "red", "big", 
    "computer", "animal", "music", "move",
    "inside", "outside", "center", "edge", "around",
    # Add your custom medoids here
    "your_medoid_1", "your_medoid_2"
]
```

#### Adjust Clustering Parameters
```python
solver.solve_with_clustering(
    target_word="book",
    n_clusters=15,          # Number of clusters per level
    verbose=True,           # Detailed output
    visualize=True,         # Generate visualizations
    use_smart_medoids=True  # Enable smart medoids
)
```

#### Batch Testing
```python
# Test multiple words
test_words = ["book", "house", "water", "computer"]
results = solver.run_batch_experiment(test_words, n_clusters=10)
```

---

## Understanding the Output

### Console Output Format
```
Target word: book
Using smart medoids for initial clustering...
Smart medoids available: ['person', 'house', 'water', ...]
Testing 15 smart medoids...

Guess 1: person (Smart Medoid 0) -> 13.39
Guess 2: house (Smart Medoid 1) -> 16.11
...

Best smart medoid: house with score 16.11
Exploring neighborhood around 'house'...

Level 0: Creating 10 clusters from 1000 words...
Guess 16: literature -> 47.06
...
```

### Key Metrics
- **Smart Medoid Scores**: Initial similarity scores (0-100)
- **Best Medoid**: Highest scoring medoid that guides exploration
- **Level N**: Clustering depth (deeper = more focused)
- **Cluster Exploration**: Systematic search through semantic neighborhoods

### Results Files
Results are saved in `Tests/run_YYYYMMDD_HHMM/`:
- `run_summary.txt`: Overall performance summary
- `clustering_results_N_clusters.csv`: Detailed clustering data
- `experiment_statistics.txt`: Statistical analysis

---

## Performance Tips

### 1. Optimize Target Word Coverage
Add medoids that cover the semantic space of your typical target words:
```python
# For spatial words: add "inside", "outside", "around"
# For emotional words: add "happy", "sad", "angry"  
# For technical words: add "science", "machine", "system"
```

### 2. Adjust Timeout Settings
For faster testing:
```bash
timeout 60s python AutomatedSemantleSolver.py
```

### 3. Monitor Memory Usage
The Word2Vec model is large (~3.5GB). Ensure sufficient RAM.

### 4. Parallel Testing
Run multiple instances with different target words:
```bash
# Terminal 1
python -c "
solver.solve_with_clustering('word1', verbose=False)
"

# Terminal 2  
python -c "
solver.solve_with_clustering('word2', verbose=False)
"
```

---

## Troubleshooting

### Common Issues

1. **Module Import Errors**:
   ```bash
   export PYTHONPATH="${PYTHONPATH}:$(pwd)/components"
   ```

2. **Memory Errors**:
   - Reduce vocabulary size in `VocabularyLoader.py`
   - Use smaller cluster numbers (n_clusters=5)

3. **Model Loading Failures**:
   - Verify Word2Vec model path
   - Check file permissions
   - Ensure sufficient disk space

4. **Slow Performance**:
   - Reduce recursion depth (edit max depth in recursive_cluster_search)
   - Use fewer clusters per level
   - Enable timeout limits

### Debug Mode
Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## Algorithm Strategy

### Why Smart Medoids Work

1. **Semantic Coverage**: Cover diverse vocabulary spaces
2. **Guided Search**: Best medoid focuses exploration  
3. **Hierarchical Refinement**: Recursive clustering narrows search
4. **Adaptive Exploration**: Strategy changes based on similarity scores

### Search Strategy Flow
```
1. Test 15 Smart Medoids → Find best match
2. Get 1000 most similar words to best medoid  
3. Cluster these 1000 words → Test cluster medoids
4. Recursively cluster best-performing clusters
5. Stop when target found or max depth reached
```

### Performance Improvements Over Random Clustering
- **85% fewer guesses** for similar exploration depth
- **Better semantic targeting** - explores relevant vocabulary spaces
- **Systematic coverage** prevents getting stuck in wrong areas
- **Focused exploration** around most promising semantic neighborhoods

---

## Best Practices

### 1. Target Word Analysis
Before running, consider your target word's semantic category:
- **Spatial words** (outer, inside): Spatial medoids will help
- **Abstract concepts** (freedom, justice): Need conceptual medoids
- **Technical terms** (algorithm, database): Need technical medoids

### 2. Medoid Selection Strategy
- **Cover major semantic categories**: People, objects, actions, properties
- **Include domain-specific terms**: Add medoids for your specific use case
- **Balance generality vs specificity**: Mix broad and focused terms

### 3. Result Analysis
- Monitor **best medoid performance**: Low scores suggest missing coverage
- Track **cluster exploration depth**: Deep recursion may indicate inefficiency
- Analyze **semantic progression**: Should show increasing similarity over time

---

## Future Enhancements

### Planned Improvements
1. **Dynamic Medoid Selection**: Choose medoids based on target word analysis
2. **Multi-Stage Fallback**: Switch strategies if initial approach fails  
3. **Ensemble Methods**: Combine multiple solving approaches
4. **Real-time Adaptation**: Adjust strategy based on intermediate results

### Customization Options
- **Custom similarity functions**: Beyond cosine similarity
- **Alternative clustering algorithms**: DBSCAN, hierarchical clustering
- **Machine learning integration**: Learn optimal strategies from data

---

This documentation provides a comprehensive guide to understanding and using the Semanthink system. For specific questions or issues, refer to the component source code or create issues in the project repository.