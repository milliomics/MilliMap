# 🧬 Spatial Omics Viewer

A lightweight, interactive 3D viewer for spatial omics data with advanced filtering, analysis tools, and beautiful visualizations.

![Milliomics Logo](Designs/milliomics_logo.png)

## 🔬 Overview

The Spatial Omics Viewer is a PyQt5-based application designed for researchers working with spatial transcriptomics and other spatial omics data. It provides an intuitive interface for exploring large-scale single-cell datasets in 3D space with real-time filtering and analysis capabilities.

## ✨ Key Features

### 🎯 **Data Visualization**
- **3D Interactive Plots**: Explore your data in true 3D using PyVista
- **Multiple Color Schemes**: Choose from Plotly/D3, Seaborn, Custom Turbo, or Milliomics brand colors
- **Real-time Filtering**: Filter by clusters, sections, and gene expression
- **Hover Information**: Get detailed cell information on mouse hover

### 🔍 **Advanced Filtering**
- **Cluster Selection**: Multi-select cluster filtering with search
- **Section Navigation**: Browse through tissue sections with up/down controls
- **Gene Expression**: Filter cells by expression levels with customizable thresholds
- **Boolean Logic**: Combine multiple filters (cells must express ALL selected genes)

### 📊 **Analysis Tools**
- **Polygon Selection**: Draw custom regions for analysis
- **Circle Selection**: Select cells within a radius
- **Point Selection**: Analyze individual cells
- **Composition Analysis**: Automatic cell type composition plots

### 🎨 **Customization**
- **Color Schemes**: Multiple professional color palettes
- **Opacity Control**: Adjust point transparency
- **Camera Persistence**: Maintain view angles during filtering
- **Search Functions**: Quick search across all lists

## 🚀 Installation

### Prerequisites
```bash
# Using conda (recommended)
conda create -n spatial-omics python=3.9
conda activate spatial-omics
```

### Dependencies
```bash
pip install -r requirements.txt
```

**Core Requirements:**
- `PyQt5` - GUI framework
- `pyvista` - 3D visualization
- `pyvistaqt` - PyVista Qt integration
- `anndata` - Single-cell data format
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `matplotlib` - Plotting
- `seaborn` - Statistical visualization
- `scipy` - Scientific computing

## 📁 Project Structure

```
milliomics/
├── README.md
├── requirements.txt
├── spatial_omics_viewer.py    # Main application file
├── data/                      # Example datasets
│   ├── codebook_0_BBSvF20210811_codebook.csv
│   ├── wholebrain_combined/
│   └── wholebrain_gene/
├── Designs/                   # Logo and brand assets
│   ├── milliomics_logo.png
│   ├── cakeinvert.png
│   └── ...
└── docs_commercial/           # Documentation
```

## 🎮 Usage

### Quick Start
```bash
# Method 1: Direct execution
python spatial_omics_viewer.py

# Method 2: As module (if organized as package)
python -m milliomics.main
```

### Loading Data
1. **Drag & Drop**: Simply drag an `.h5ad` file onto the application window
2. **File Dialog**: Click "Load File" to browse and select your data
3. **Gene Data**: Use "Load Gene Data" for additional gene-level information

### Data Format Requirements
Your AnnData object should contain:
- `obsm['spatial']`: Spatial coordinates (n_cells × 2 or 3)
- `obs['clusters']`: Cell type annotations (optional)
- `obs['source']`: Section/sample information (optional)
- Gene expression matrix in `.X`

### Example Usage
```python
import anndata as ad
import numpy as np

# Load your spatial omics data
adata = ad.read_h5ad("your_data.h5ad")

# Ensure spatial coordinates exist
# adata.obsm['spatial'] = coordinates  # shape: (n_cells, 2) or (n_cells, 3)

# Add cluster information
# adata.obs['clusters'] = cluster_labels

# Launch viewer
from spatial_omics_viewer import main
main()
```

## 🎨 Color Schemes

### Available Palettes
- **🎨 Plotly/D3**: Professional, up to 60 distinct colors
- **🌈 Custom Turbo**: Vibrant, high-contrast colors
- **📊 Seaborn**: Scientifically optimized palettes
- **🎂 Milliomics**: Brand colors (Pink, Green, Gray)
- **📁 AnnData**: Use original colors from your data

### Usage
```python
# Colors are automatically assigned based on unique clusters
# Manually access color functions:
from colors import generate_milliomics_palette
colors = generate_milliomics_palette(n_clusters=10)
```

## 🔬 Analysis Workflows

### 1. Exploratory Analysis
1. Load your dataset
2. Browse through sections using up/down navigation
3. Filter by clusters of interest
4. Examine gene expression patterns

### 2. Region-Specific Analysis
1. Enable "Analysis Mode"
2. Use polygon selection to define regions
3. View automatic composition analysis
4. Export results for further analysis

### 3. Gene Expression Analysis
1. Select genes of interest
2. Set expression thresholds
3. Toggle "Gene Only" mode for focused visualization
4. Compare expression across regions

## 🛠️ Advanced Features

### Memory Management
- Optimized for large datasets (tested with >1M cells)
- Efficient filtering using boolean masks
- Smart rendering with opacity controls

### Performance Tips
- Use section filtering for large multi-section datasets
- Limit gene lists to ~20,000 for UI responsiveness
- Adjust point opacity for better performance with dense data

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
git clone https://github.com/your-username/milliomics.git
cd milliomics
conda create -n milliomics-dev python=3.9
conda activate milliomics-dev
pip install -r requirements.txt
```

## 📝 Citation

If you use this software in your research, please cite:

```bibtex
@software{spatial_omics_viewer,
  title={Spatial Omics Viewer: Interactive 3D Visualization for Spatial Transcriptomics},
  author={Milliomics Team},
  year={2024},
  url={https://github.com/your-username/milliomics}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🐛 Bug Reports & Feature Requests

Please use the [GitHub Issues](https://github.com/your-username/milliomics/issues) page to:
- Report bugs
- Request new features
- Ask questions about usage

## 💬 Support

- **Documentation**: Check this README and inline code comments
- **Issues**: GitHub Issues page
- **Email**: [your-email@institution.edu](mailto:your-email@institution.edu)

## 🏆 Acknowledgments

- Built with [PyVista](https://pyvista.org/) for 3D visualization
- Uses [AnnData](https://anndata.readthedocs.io/) format for single-cell data
- Color palettes inspired by [Plotly](https://plotly.com/) and [Seaborn](https://seaborn.pydata.org/)
- Special thanks to the spatial transcriptomics community

---

**Milliomics** - *Making spatial omics accessible to everyone* 🧬✨ 