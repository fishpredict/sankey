# Sankey Species Visualization Application

Interactive application for visualizing fish species sequences using Sankey diagrams.

## Features

- **Interactive visualization**: Sankey diagrams to explore species sequences
- **Site filtering**: Selection of specific sampling sites
- **Complexity control**: Adjustable maximum number of species (1-15)
- **HTML export**: Save visualizations in interactive format
- **Responsive interface**: Modern design with Bootstrap
- **Memory monitoring**: Built-in memory usage tracking

## Quick Start

### Prerequisites

- Python 3.9 or higher
- Virtual environment (recommended)

### Installation

1. **Create and activate virtual environment**
```bash
# Create virtual environment (use python3 on macOS)
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
# Option 1: With activated virtual environment
python app.py

# Option 2: Direct execution (recommended for macOS)
./venv/bin/python app.py

# Option 3: If python command not found, use python3
python3 app.py
```

The application will be available at: http://localhost:7860

## Usage

1. **Select a site** from the dropdown menu
2. **Adjust the number of species** using the slider (1-15)
3. **Click "Update Diagram"**
4. **Explore** with zoom, pan, etc.
5. **Export to HTML** if needed

## Project Structure

```
sankey-docker-v1/
├── app.py                    # Main application file
├── requirements.txt          # Python dependencies
├── README.md                # This file
├── data/                    # Data directory
│   ├── dict_id2species.txt  # Species dictionary (JSON format)
│   └── gpt2_med_env30_abu100_240ep_final_model_500inf_test_dataset.csv
└── venv/                    # Virtual environment (created after installation)
```

## Technologies

- **Backend**: Python 3.9, Dash, Plotly
- **Data Processing**: Pandas, NumPy
- **Frontend**: Bootstrap, HTML5
- **Monitoring**: psutil for memory tracking

## Data

The application uses GPT-2 model predictions for Australian fish species abundance:
- **Sites**: 702 sampling sites
- **Species**: 1158 different species
- **Observations**: 351,000 data points

### Data Files

- `dict_id2species.txt`: JSON dictionary mapping species IDs to species names
- `gpt2_med_env30_abu100_240ep_final_model_500inf_test_dataset.csv`: Main dataset containing:
  - Sample information (sample_num, sample_index)
  - Species abundance data (SP1, SP2, ..., SP1158)
  - Additional metadata columns

## Troubleshooting

### Application won't start

```bash
# Check dependencies
pip list | grep dash

# Check if port is available
lsof -i :7860

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Data loading issues

```bash
# Check data files exist
ls -la data/

# Check file sizes
du -h data/*.csv
du -h data/*.txt

# Test data loading
python -c "import pandas as pd; print(pd.read_csv('data/gpt2_med_env30_abu100_240ep_final_model_500inf_test_dataset.csv').shape)"
```

### Memory issues

The application includes built-in memory monitoring. If you encounter memory issues:

1. Reduce the number of species in the visualization (use the slider)
2. Select fewer sites for analysis
3. Monitor memory usage in the console output

### Virtual environment issues

```bash
# Remove and recreate virtual environment (macOS/Linux)
rm -rf venv/
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Alternative: Direct execution without activation
./venv/bin/python app.py
```

### Common macOS Issues

**Problem**: `zsh: command not found: python`
**Solution**: Use `python3` instead of `python` or use direct path to virtual environment Python:
```bash
# Use python3
python3 -m venv venv
python3 app.py

# Or use direct path
./venv/bin/python app.py
```

**Problem**: Virtual environment activation issues
**Solution**: Use direct execution:
```bash
cd /path/to/GitLab_Sankey_App
./venv/bin/python app.py
```



