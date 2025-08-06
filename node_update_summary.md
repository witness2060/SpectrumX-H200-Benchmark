# Node Name Update Summary

This document lists all files that contain old node names (node001-009) that need to be updated to the new format (fukushimadc-02-hgx-XXXX).

## Files Requiring Updates:

### 1. Shell Scripts
- **scripts/parallel_gpu_test.sh**
  - Lines 19, 22, 25: Contains hardcoded node lists
  - Old: `node001,node002`, `node001,node002,node003,node004`, etc.
  - New: Should use dynamic node detection from cluster_config.sh

- **setup/setup_slurm.sh**
  - Lines 58-65: NodeName entries for Slurm configuration
  - Lines 68: PartitionName with node range
  - Lines 75-82: GPU configuration entries
  - Old: `node001-008` format
  - New: `fukushimadc-02-hgx-0001` through `fukushimadc-02-hgx-0009`

- **setup/install_python_packages.sh**
  - Line 122: Help text example
  - Old: `node001,node002`
  - New: `fukushimadc-02-hgx-0001,fukushimadc-02-hgx-0002`

### 2. Python Scripts
- **scripts/generate_report.py**
  - Line 107: Documentation comment
  - Old: `node001-007, node009`
  - New: `fukushimadc-02-hgx-0001 through 0007, 0009`

### 3. Documentation Files
- **README.md**
  - Line 116: Example export command
  - Line 227: SSH connection example
  - Old: `node001,node002,node003,node004`
  - New: `fukushimadc-02-hgx-0001,fukushimadc-02-hgx-0002,fukushimadc-02-hgx-0003,fukushimadc-02-hgx-0004`

### 4. Other Documentation (Lower Priority)
The following documentation files contain references but may be historical/archival:
- PROJECT_STATUS_REPORT.md
- docs/SPECTRUMX_H200_EXECUTION_GUIDE.md
- FIXES_APPLIED.md
- PROJECT_ANALYSIS_REPORT.md
- docs/FINAL_ACHIEVEMENT_REPORT.md
- docs/performance_improvement_report.md
- docs/final_performance_report.md
- docs/final_benchmark_report.md

## Note
The cluster_config.sh file already contains the correct new node names and provides dynamic node detection functionality. Other scripts should be updated to use this configuration rather than hardcoding node names.