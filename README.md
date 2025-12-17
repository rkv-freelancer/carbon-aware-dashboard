# Carbon-Aware Dashboard

A comprehensive monitoring and visualization dashboard for tracking energy consumption and carbon emissions of machine learning model training across different cloud regions.

## Overview

This project provides real-time monitoring of GPU, CPU, and memory usage during ML model fine-tuning, with a focus on carbon-aware computing. It tracks energy consumption and correlates it with regional carbon intensity data to help make environmentally conscious decisions about where to train models.

## Features

- **Real-time Monitoring**: Track GPU, CPU, DRAM, and memory metrics during model training
- **Carbon Awareness**: Integrate regional carbon intensity data to calculate environmental impact
- **Multi-Region Support**: Compare training across US West, East, and Central regions
- **Interactive Dashboard**: Visualize metrics using Plotly/Dash
- **Modal Integration**: Seamless deployment on Modal's infrastructure with GPU support
- **Volume Persistence**: Save monitoring data (CSV/JSON) for later analysis

## Project Structure

```
carbon-aware-dashboard/
├── WattsOnAI/           # Core monitoring library
├── demo.py              # Simple dashboard demo
├── demo_revised.py      # Full training & monitoring implementation
└── README.md
```

## Prerequisites

- Python 3.12+
- Modal account ([modal.com](https://modal.com))
- Environment variables for secrets (`.env` file)
- NVIDIA GPU access (for training)

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd carbon-aware-dashboard
```

2. Install Modal CLI:
```bash
pip install modal
```

3. Authenticate with Modal:
```bash
modal setup
```

4. Create `.env` file with required secrets

## Usage

### Running the Dashboard (Local)

View pre-existing monitoring data:

```bash
python demo.py
```

### Training Models with Monitoring

Train models across regions with full monitoring:

```bash
modal run demo_revised.py
```

### Testing Components

Test carbon intensity regions:
```bash
modal run demo_revised.py::test_carbon_regions
```

Test GPU class configuration:
```bash
modal run demo_revised.py::test_gpu_class
```

Diagnose monitoring capabilities:
```bash
modal run demo_revised.py::diagnose_monitoring
```

### Viewing Dashboard (Modal-hosted)

Access the interactive dashboard:
```bash
modal serve demo_revised.py
```

Then visit the URL provided by Modal.

### Resetting Data

Clear volume and restart tracking:
```bash
modal run demo_revised.py::reset
```

## Model Configurations

Currently supports:
- **FLAN-T5 Small** (~77M parameters)
- **BART Small** (~70M parameters)

Regions tested:
- US West (Oregon, California)
- US East (Virginia)
- US Central (Texas, Iowa)

## Monitoring Metrics

The dashboard tracks:
- **Energy**: CPU, DRAM, GPU (Joules → Watts conversion)
- **Performance**: Training time, tokens/second
- **Carbon**: Regional carbon intensity (gCO2/kWh)
- **Resources**: Memory usage, GPU utilization

## Data Output

Monitoring data saved to Modal volume:
```
/vol/monitoring_data/
├── csv/              # Time-series metrics
└── json/             # Aggregated summaries
```

## Configuration

Key settings in `demo_revised.py`:
- `GPU_TYPE`: GPU specification (default: A100-40GB)
- `TEST_REGIONS`: Regions to evaluate
- `TIMEOUT_HOURS`: Maximum training duration
- `MAX_RETRIES`: Retry attempts on preemption

## Dependencies

Core libraries:
- `transformers` - Hugging Face models
- `WattsOnAI` - Energy monitoring
- `dash`/`plotly` - Visualization
- `modal` - Cloud infrastructure
- `psutil` - System monitoring

## Contributing

Contributions welcome! Areas for improvement:
- Additional model architectures
- More cloud regions
- Enhanced visualizations
- Optimization recommendations

## License

[Specify your license]

## Acknowledgments

- Modal for cloud infrastructure
- WattsOnAI for energy monitoring capabilities
- Hugging Face for model access
