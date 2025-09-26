# Sankey Species Visualization Application

Interactive application for visualizing fish species sequences using Sankey diagrams.

## Installation

### Prerequisites

- Python 3.9 or higher
- Git LFS installed on your system

### Step 1: Install Git LFS

**macOS:**
```bash
brew install git-lfs
```

**Ubuntu/Debian:**
```bash
sudo apt-get install git-lfs
```

**Windows:**
Download and install from: https://git-lfs.github.io/

Initialize Git LFS:
```bash
git lfs install
```

### Step 2: Clone the Repository

```bash
git clone https://github.com/fishpredict/sankey.git
cd sankey
```

### Step 3: Download Large Files

```bash
git lfs pull
```

This downloads the 1.2GB dataset file required for the application.

### Step 4: Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Run the Application

```bash
python app.py
```

The application will be available at: http://localhost:7860

## Usage

1. Select a site from the dropdown menu
2. Adjust the number of species using the slider (1-15)
3. Click "Update Diagram"
4. Explore with zoom, pan, etc.
5. Export to HTML if needed

## Project Structure

```
sankey/
├── app.py                    # Main application file
├── requirements.txt          # Python dependencies
├── README.md                # This file
├── data/                    # Data directory
│   ├── dict_id2species.txt  # Species dictionary
│   └── gpt2_med_env30_abu100_240ep_final_model_500inf_test_dataset.csv
└── .gitattributes           # Git LFS configuration
```

## Troubleshooting

### Git LFS files not downloaded

```bash
git lfs install
git lfs pull
```

### Application won't start

```bash
# Check dependencies
pip install -r requirements.txt

# Check data files exist
ls -la data/

# Verify large file was downloaded
du -h data/*.csv
```

### Memory issues

The application processes a 1.2GB dataset. If you encounter memory issues:

1. Reduce the number of species in the visualization
2. Select fewer sites for analysis
3. Ensure you have at least 4GB of available RAM

### Virtual environment issues

```bash
# Remove and recreate virtual environment
rm -rf venv/
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### macOS specific issues

If `python` command is not found, use `python3`:

```bash
python3 -m venv venv
python3 app.py
```
