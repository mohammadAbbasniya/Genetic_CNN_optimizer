# CNN Optimization using Genetic Algorithm
This repository contains an implementation of a Genetic model that tries to optimize the structure of a Convolution Neural Network (CNN).  

### Requirements
only `tensorflow` is needed. See [requirements.txt](requirements.txt) and use `pip install -r requirements.txt` for installation.
<br><br>

## About Genetic-algorithm
In computer science and operations research, a genetic algorithm (GA) is a metaheuristic inspired by the process of natural selection that belongs to the larger class of evolutionary algorithms (EA). Genetic algorithms are commonly used to generate high-quality solutions to optimization and search problems by relying on biologically inspired operators such as mutation, crossover and selection. Some examples of GA applications include optimizing decision trees for better performance, solving sudoku puzzles, hyperparameter optimization, causal inference, etc. [[Wiki](https://en.wikipedia.org/wiki/Genetic_algorithm)].
<p align='center'>
  <img height="300" alt="genetic-example" src="README.imgs/genetic-example.jpg">
</p>



## Project structure
- ### 📄 file [main.ipynb]
  This  file is my Demo file that shows how to use this classifier on `cifar10` dataset (provided by `keras.datasets`).
  
- ### 📂 directory [genetic_cnn_optimizer]
    - #### 📄 file [cnn_chromo.py]
      This file contains `CNNChromo` class. This is the representation of a CNN in genetic. Each chromosome consists of an array:
      <p align='center'>
          <img height="200" alt="genetic-example" src="https://github.com/mohammadAbbasniya/Genetic_CNN_optimizer/blob/main/README.imgs/chromo.png">
      </p>
      I have implemented two Crossover operations, the first one is a pivot-based crossover and the other uses a random binary list for dividing genomes.
      <p align='center'>
          <img height="200" alt="genetic-example" src="https://github.com/mohammadAbbasniya/Genetic_CNN_optimizer/blob/main/README.imgs/crossover.png">
      </p>
      The mutation operation will change one genome randomly. There are three conditions that may happen in mutation. <b>(a)</b>change one genome $L_{i}$ or  $K_{i}$ &nbsp; <b>(b)</b>increase the number of layers &nbsp; <b>(c)</b>decrease the number of layers 
      <p align='center'>
          <img height="130" alt="genetic-example" src="https://github.com/mohammadAbbasniya/Genetic_CNN_optimizer/blob/main/README.imgs/mutate.png">
      </p>

    - #### 📄 file [genetic_cnn.py]
      This file contains `GeneticCNN_finder` class. This is where Genetic Algorithm is implemented. The constructor of this class takes a parameter `accuracy_function` that should be a function for evaluating the accuracy of every CNN created by the genetic. 

