# Covid-Connectome



**Purpose:** Pipelines and helper scripts used in my thesis **"Cambios estructurales del cerebro en pacientes Covid por análisis de imágenes de resonancia magnética"** to compare MRI-derived brain networks between a **Control** group and a **Covid-19** (Treatment) group.
**Author:** Alejandro Garma Oehmichen  

---

## Pipeline overview

| **Input modalities** | **Key outputs** |
|----------------------|-----------------|
| • T1-weighted (T1w)  <br>• Diffusion-weighted imaging (DWI)  <br>• Resting-state fMRI (rs-fMRI) | • Structural connectomes  <br>• Functional connectomes  <br>• Network-Based Statistics (NBS) significant clusters  <br>• Group-level statistical tests |

> **Important:** The workflow is **not fully automated yet**.  
> You will still need to run certain steps manually in **CONN** (functional connectivity) and **NBS** (cluster significance).

---

## Repository structure

~~~text
.
├── fsl/                 # FSL configs (e.g. eddy, design.fsf)
├── python_analysis/     # Python utilities for statistics & ML
├── scripts/             # Bash wrappers for MRtrix, FreeSurfer, ANTs, SynBo-Disco
│   └── matlab/          # MATLAB helpers for CONN & NBS outputs
└── utils/               # Atlas files and misc resources
~~~

### Folder descriptions

| Directory | Contents |
|-----------|----------|
| **fsl** | Config files for: <ul><li>running *eddy* on DWI data</li><li>rs-fMRI preprocessing (`design.fsf`)</li></ul> |
| **python_analysis** | High-level analysis scripts: <ul><li>group statistics</li><li>Gradient Boosting on structural connectomes</li></ul> |
| **scripts** | Shell pipelines for MRtrix, FreeSurfer, ANTs, SynBo-Disco. Includes MATLAB scripts to parse CONN results and export significant NBS clusters. |
| **utils** | Lookup tables and atlas files required by the above pipelines. |

---

## Getting started

1. **Clone the repo**

   ~~~bash
   git clone https://github.com/Garmanta/Covid-Connectome.git
   git clone 
   cd Covid-Connectome
   ~~~

2. **Set up environments**

   | Software   | Version (tested) | Notes |
   |------------|------------------|-------|
   | FSL        | ≥ 6.0            | Required for *eddy* and FEAT |
   | MRtrix3    | 3.0.4            | Tractography & structural connectomes |
   | FreeSurfer | ≥ 7.3            | Cortical parcellation |
   | ANTs       | ≥ 2.4            | Image registration |
   | CONN       | 23b              | GUI step for functional connectivity |
   | NBS        | 1.2              | GUI/CLI step for network statistics |
   | Python     | ≥ 3.9            | Statistical testing |
   | MATLAB     | R2022a           | Only for helper scripts (optional) |

2. **Getting started**

   Set up your BIDS image database. Then run preprocessing.sh in the main directory and follow the comments instructions in each script. 
---

*Last updated: 2025-05-15*
