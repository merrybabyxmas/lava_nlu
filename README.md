# LAVA

## Installation

### Prerequisites

Before installing LAVA, you need to set up the PEFT_LAVA dependency:

```bash
# Clone the PEFT_LAVA repository
git clone https://github.com/merrybabyxmas/peft_lava.git peft
cd peft

# Run the setup script to install PEFT_LAVA
./setup.sh

# The script will:
# - Create a conda environment named 'lava' with Python 3.10.19
# - Install PEFT_LAVA in editable mode
# - Set up symbolic links for proper package resolution
# - Verify the installation

# Return to your working directory
cd ..
```

### Install LAVA

```bash
# Clone the LAVA repository
git clone https://github.com/merrybabyxmas/lava.git
cd lava

# Activate the lava environment (created by PEFT_LAVA setup)
conda activate lava

# Install LAVA dependencies
pip install -e .
```

## Usage

Make sure to activate the conda environment before using LAVA:

```bash
conda activate lava
```