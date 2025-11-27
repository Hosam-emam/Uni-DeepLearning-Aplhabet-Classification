# Project Setup Guide

## Overview
This project involves visualizing datasets for Arabic and English characters. The datasets are loaded from CSV files and visualized using Python libraries such as `matplotlib` and `pandas`.

## Prerequisites
Ensure you have the following installed on your system:
- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Set Up a Virtual Environment** (Optional but recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install Required Packages**
   Install the dependencies listed in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Structure
The datasets are organized as follows:
```
Data/
├── Arabic/
│   ├── csvTestImages.csv
│   ├── csvTestLabel.csv
│   ├── csvTrainImages.csv
│   ├── csvTrainLabel.csv
│   ├── Test Images 3360x32x32/
│   │   └── test/
│   ├── Train Images 13440x32x32/
│       └── train/
├── English/
    ├── emnist-balanced-test.csv
    ├── emnist-balanced-train.csv
```

## Running the Notebook

1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

2. Navigate to the `raw_data.ipynb` file and open it.

3. Run the cells sequentially to load the datasets, process the data, and visualize the results.

## Notes
- Ensure the dataset files are in the correct paths as specified in the notebook.
- If you encounter any missing library errors, install them using pip.

## Contact
For any issues or questions, please contact [Your Name] at [Your Email].
