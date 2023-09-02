# Quantum-Assisted Generative Algorithms
## Introduction
This repository hosts the resources for my Bachelor's thesis in Computer Science. The research was conducted at the Department of Mathematics at Imperial College London and was defended at the Escuela Politécnica Superior of the Autonomous University of Madrid. The primary aim of this work is to leverage the principles of quantum computing to enhance machine learning algorithms. The immediate focus is on improving the learning of one-dimensional distributions using Generative Adversarial Networks (GANs) through a novel approach: Quantum-Assisted Associative Adversarial Networks (QAAANs). The broader goal is to explore the transformative possibilities that quantum computing brings to the machine learning landscape.



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

## Dependencies

This project relies on a wide range of Python libraries for tasks like machine learning, statistical analysis, data visualization, and quantum computing. The principal dependencies include:

- `tensorflow` and `keras`: For machine learning models, particularly Generative Adversarial Networks and Restricted Boltzmann Machines.
- `numpy`: For numerical computations and data manipulation.
- `scipy`: Specifically employed for statistical functions like `wasserstein_distance` and statistical distributions such as `norm`, `uniform`, `cauchy`, `pareto`, and `beta`.
- `matplotlib`: For generating plots and other data visualizations. `GridSpec` is also used for customizing the layout of multiple subplots.
- `json` and `os`: For file handling and operating system related tasks.
- `time`: For timing code execution for performance analysis.
- `pyqubo`: For generating Binary Quadratic Models (BQM).
- `dwave`: For solving optimization problems on the D-Wave quantum annealer.
- `mnist` dataset from `tensorflow.keras.datasets`: For testing machine learning models on image data.



## Getting Started

### Initial Setup
To get started, clone this repository to your local machine. It's advisable to have Python installed, preferably via an Anaconda distribution to ensure all dependencies are met. After cloning, navigate to the `src` directory where the main source code resides.

### Configuration and Running the Code
The code is designed to be flexible with both hyperparameters and distribution parameters. You can adjust the model hyperparameters by modifying the corresponding hyperparameter dictionary in the main function for each script. Additionally, you can specify the probability distribution to be learned and its corresponding parameters. You can also use the alternative `main` functions specifically designed for hyperparameter testing and testing different probability distributions in the same run.

#### Classical Models
To run classical models like Vanilla GAN, MMD GAN, or Classical RBM, execute the respective Python files:
- `a_vanilla_gan_1d.py`
- `b_mmd_gan_1d.py`
- `c_classical_rbm.py`

#### Quantum Models
For models that require interaction with a quantum computer, such as:
- `d_quantum_rbm.py`
- `e_vanilla_qaaan.py`
- `f_mmd_qaaan.py`

You'll need to configure access to Leap’s solvers. Detailed instructions can be found at the [D-Wave Leap documentation](https://docs.ocean.dwavesys.com/en/stable/overview/sapi.html#create-a-configuration-file).

### Further Documentation
If you encounter any issues or seek deeper insights into the project, please refer to the accompanying thesis, available in this repository as `thesis.pdf`. This comprehensive document elaborates on the theoretical framework and empirical results.



## Contributing

If you wish to contribute to this project, please:

1. Fork the repository.
2. Clone your fork.
3. Create a new branch to work on.
4. Commit your changes.
5. Submit a pull request.

You can also open an issue to discuss what you would like to add or modify in the project.

## Acknowledgements

I would like to extend my deepest appreciation to my thesis advisor, Dr. Jack Jacquier, of the Department of Mathematics at Imperial College London. His unwavering support, expert guidance, and constructive criticism have been instrumental in shaping this research from its inception to completion.

I am equally grateful to my second advisor, Alberto Suárez at the Autonomous University of Madrid, for his invaluable comments and insights on the thesis. His input has been a crucial component in ensuring the rigor and quality of this work.

My sincere thanks also go to the Department of Mathematics at Imperial College London and the Escuela Politécnica Superior at the Autonomous University of Madrid for providing the resources and environment that have made this research possible.

Finally, my gratitude extends to all those who have indirectly contributed to this project. This includes the developers and maintainers of the open-source tools and libraries that have been vital to this research.


