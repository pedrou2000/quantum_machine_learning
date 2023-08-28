# Quantum-Assisted Generative Algorithms
## Introduction
This repository hosts the resources for my Computer Science thesis, which primarily aims to leverage the principles of quantum computing to enhance machine learning algorithms. While the immediate focus is on improving the learning of one-dimensional distributions using Generative Adversarial Networks (GANs) through Quantum-Assisted Associative Adversarial Networks (QAAANs), the broader goal is to explore the transformative possibilities that quantum computing brings to the machine learning landscape.


## Thesis Overview
The thesis is fundamentally designed to explore the implications of merging quantum computing principles with Generative Adversarial Networks, crystallized in the form of Quantum-Assisted Associative Adversarial Networks (QAAANs). The primary drive is to improve learning of one-dimensional distributions while examining the pros and cons of classical and quantum models.

The research journey starts with a 'State of the Art' section that sets the academic background, covering essentials of Quantum Computing, Quantum Annealing, Boltzmann Machines (BMs), and GANs. This section also delves into recent developments in Quantum Machine Learning, forming the theoretical base for the subsequent discussions.

The 'Development' section encapsulates the core technicalities, detailing the development and implementation of both classical and quantum models. It specifically shines a light on the architectural and computational choices made for GANs, RBMs in the Quantum Annealer, and QAAANs.

Following this, the 'Experimental and Results' section offers a comprehensive exposition of experiments performed and their findings. It captures a comparison of the effectiveness of classical and quantum models, with an eye on their performance across various distributions.

The thesis concludes by offering reflective 'Conclusions', covering the project’s key outcomes, potential areas for improvement, and avenues for future research.


## Repository Structure

This repository is organized into distinct directories and files, each serving a specific function in the context of the thesis.

### Thesis Document
Located in the root directory as `thesis.pdf`.

### Source Code
`/src/`  
- **Source Code Files**: Houses the Python source code files crucial to the implementation of the project.
  - `a_vanilla_gan_1d.py`: Implements Vanilla GAN for learning one-dimensional distributions.
  - `b_mmd_gan_1d.py`: Code for MMD GAN tailored to one-dimensional distributions.
  - `c_classical_rbm.py`: Implements the Classical Restricted Boltzmann Machine.
  - `d_quantum_rbm.py`: Source code for the Quantum Restricted Boltzmann Machine.
  - `e_vanilla_qaaan.py`: Code for Quantum Vanilla QAAAN.
  - `f_mmd_qaaan.py`: Source file for Quantum MMD QAAAN.

### Results
`/results/`  
- **Experimental Outputs**: Contains all final results from the experiments conducted, along with graphical representations for better comprehension.

### Table Generation Scripts
`/src/`  
- `g_table_generation_vertical.py`: Script for generating vertical tables based on the results.
- `h_table_generation_horizontal.py`: Script for creating horizontal tables summarizing the experimental data.

### Python Package Initialization
`__init__.py`  
- **Initialization File**: Serves as the initialization file for Python packages, enabling package-level variables and methods.


## Getting Started

To clone this repository, execute the following command in your terminal:

```bash
git clone <repository_url>
```

For setting up the environment and executing the code, please refer to the README within the `/src/` directory.

## Dependencies

The project relies on the following dependencies:

- Python 3.x
- NumPy
- D-Wave Ocean SDK
- TensorFlow or PyTorch
- Additional libraries as required

## Contributing

If you wish to contribute to this project, please:

1. Fork the repository.
2. Clone your fork.
3. Create a new branch to work on.
4. Commit your changes.
5. Submit a pull request.

You can also open an issue to discuss what you would like to add or modify in the project.

## Acknowledgments

Heartfelt thanks to my advisors and everyone who contributed to the open-source packages utilized in this research.
