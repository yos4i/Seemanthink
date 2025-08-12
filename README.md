# Semanthink Setup Instructions

## Windows Setup

### 1. Create Virtual Environment
```cmd
python -m venv venv
```

### 2. Activate Virtual Environment
```cmd
venv\Scripts\activate
```

### 3. Install Dependencies
```cmd
pip install -r requirements.txt
```

### Running on Windows
```cmd
venv\Scripts\activate
cd components
python AutomatedSemantleSolver.py [target_word] [options]
```

## Linux Setup

### 1. Install Python Virtual Environment Support
```bash
apt install python3.12-venv
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
```

### 3. Activate Virtual Environment
```bash
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### Running on Linux
```bash
source venv/bin/activate
cd components
python3 AutomatedSemantleSolver.py [target_word] [options]
```

## Running the Project

After completing the setup above, you can run the Python files in the components directory.

### AutomatedSemantleSolver Examples
```bash
# Use default settings (smart medoids, random target word)
python AutomatedSemantleSolver.py

# Specify target word with smart medoids (default)
python AutomatedSemantleSolver.py book

# Use smart medoids explicitly
python AutomatedSemantleSolver.py book --medoids smart

# Use random medoids with custom cluster count
python AutomatedSemantleSolver.py book --medoids random --clusters 15
```

**Command Options:**
- `target_word` (optional): The word to solve for
- `--medoids`: Choose `smart` (default) or `random` medoid strategy  
- `--clusters`: Number of clusters when using random medoids (default: 10)

### Other Components
The main automation script requires:
- Word2Vec model file (GoogleNews-vectors-negative300.bin)
- Vocabulary file (English-Words_Semantle_filtered.txt)
- Complete clustering system setup

Check the components directory for available Python modules.

