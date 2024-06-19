# A Genetic Algorithm for Feature Selection in Hierarchical Classification

**Feature selection** is a crucial preprocessing step in the realm of data mining. Its primary goal is to minimize the number of original attributes in a database, thereby optimizing the performance of a predictive model.

Despite the advantages of feature selection for classification tasks, there is a noticeable gap in the literature when it comes to attribute selection within the context of global hierarchical classification.

This project aims to bridge this gap by proposing a **evolucionary supervised feature selection approach** that leverages genetic algorithms in two distinct ways.

## Table of Contents
- [Methodologies](#Methodologies)
- [Project Structure](#Structure)
- [Objectives](#Objectives)
- [How to use](#How-to-use)
- [Contact Information](#Contact)
- [Requirements and libraries descriptions](#Requirements)

## Methodologies

### Genetic Algorithm with Global Model Naive Bayes (GAwGMNB)

The first approach involves the use of a **Genetic Algorithm** in conjunction with the **Global Model Naive Bayes (Carlos N. Silla Jr., 2009)** as a fitness function. This method is further enhanced by a multiprocessing Python tool, which facilitates parallelism and accelerates the population evaluation process conducted by the genetic algorithm.

## Structure

```sh
├── data/                   .arff hierarchical multi-label datasets.
|  ├── best_chromossome/    The output files of the best attributes selected.
|  ├── temp/                Paralellism temporary files.
|
├── docs/                   Documentation (articles, utilities, project files, etc).
├── generated_files/        Logs, train database for Neural Network and 'checkpoint' file for errors.
├── results/                Datasets results for each algorithm.
├── src/                    Source code.
|  ├── nbayes.so            Source library of Global Model Naive Bayes, compile it on your own machine in 'docs/utils'.
|
└─ __main__.py              Main file. 
```

### Genetic Algorithm with Neural Network (GAwNN)

The second approach employs a **Genetic Algorithm** with a **Neural Network** that is specifically trained for the dataset as a fitness function. Initially, we generate 'x' objects with the Global Model Naive Bayes (Carlos N. Silla Jr., 2009) for 'y' epochs. Subsequently, we train the Neural Network based on the training dataset from the input file and utilize the Neural Network as a Fitness Function.

By implementing these methodologies, this project strives to advance the field of feature selection and contribute to the body of knowledge on global hierarchical classification.

## Objectives

The main goals to propose the development of a hybrid feature selection method that takes into account the hierarchy of classes, combining correlation-based filter techniques to assess the quality of attribute subsets and genetic algorithm-based techniques to search for attribute subsets. This aims to construct solutions capable of improving the predictive performance of global hierarchical classifiers. In this regard, the specific objectives of this work are:

1. **Development of a evolucionary feature selection method**: This method needs to be:
   - Fast;
   - Scalable;
   - Capable of reducing the number of attributes in databases without compromising the classification model's performance.

2. **Experiments**: Conducting computational experiments using hierarchical databases related to protein function prediction and image problems.

3. **Analyzing the state of the art**: Comparing the proposed approach with other feature selection methods available in the literature and with the use of global classifiers without attribute selection.

## How-to-use

To get started, you'll need a .arff file for both the training and testing datasets. Ensure that these datasets are discretized following this format:

``` .arff 

@relation ExampleDataset

@ATTRIBUTE attribute1 NUMERIC
@ATTRIBUTE attribute2 NUMERIC
@ATTRIBUTE attribute3 NUMERIC
@ATTRIBUTE class {1.1, 1.2, 1.1.2}

@data
0.1, 0.2, 0.3, 1.1
0.0, 0.2, 0.2, 1.2
0.4, 0.3, 0.4, 1.1.2

```

- The **@ATTRIBUTE class** represents the hierarchy of the dataset.

The discretization is already integrated into the project to use Global Mode Naive Bayes and can be found in `src/preprocessing.py`.

Once your datasets are prepared, update the file paths in `config.yaml` to point to your one or more dataset(s) and the type of algorithm you want, (GAwGMNB) or (GAwNN). This ensures that the main script operates on the correct data. Following these steps, you'll be ready to use the project effectsively.

## Contact

For any inquiries or suggestions, please don't hesitate to contact me. You can reach me at my [e-mail](mailto:gssantoz2012@gmail.com) for more information.

Thank you for your interest!

## Requirements
Ensure the following dependencies are installed before running the project:

- **liac-arff**: You can obtain the LIAC-ARFF library from: https://github.com/renatopp/liac-arff. 
This library is essential for reading and writing ARFF files, providing a seamless interaction with ARFF-formatted datasets.

- **scikit-learn**: The project relies on scikit-learn, a widely-used machine learning library in Python. 

- **pandas**: It provides data structures like DataFrame for efficient handling of structured data. 

- **numpy**: It provides support for large, multi-dimensional arrays and matrices, along with mathematical functions to operate on these elements.

- **matplotlib**: It is employed in this project for data visualization, enabling the generation of informative plots and charts.

- **yaml**: A human-friendly data serialization standard for all programming languages. It is commonly used for configuration files and in applications where data is being stored or transmitted.

- **tensorflow**: An open-source platform for machine learning. It has a comprehensive, flexible ecosystem of tools, libraries, and community resources that lets researchers push the state-of-the-art in ML and developers easily build and deploy ML-powered applications.

- **multiprocessing**: A python module that supports spawning processes using an API similar to the threading module. It allows you to leverage multiple processors on a machine, which leads to faster execution.

- **ctypes**: A foreign function library for Python. It provides C compatible data types and allows calling functions in DLLs/shared libraries. It can be used to wrap these libraries in pure Python.
