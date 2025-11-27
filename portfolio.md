# Dhaka House Rent Predictor: A First-Principles Machine Learning Implementation

## Project Overview

This document presents the development and analysis of the project in three parts:

*   **1: The Application** details the engineering of a complete ML pipeline—from data ingestion to web serving—using only the Python standard library, enforcing a strict "Zero Dependency" philosophy.
*   **2: The Experimentation** presents a rigorous performance analysis, comparing the custom implementation against Scikit-learn. It explores how low-level optimizations in C (including manual memory management and multi-threading) can outperform established libraries.
*   **3: Limitations & Future Scope** reflects on the current system's constraints and outlines a roadmap for evolving it into a more robust, cloud-native solution.

## 1: The Application

### 1.1 Overview

This is a comprehensive machine learning system designed to predict house rents in Dhaka, Bangladesh. Unlike typical ML projects that rely on high-level frameworks, this application is built entirely from first principles using only the Python standard library.

The core philosophy is **"Zero Dependency"**. Every component—from the CSV parser and data downloader to the linear algebra engine and web server—is implemented from scratch. This approach serves as a rigorous exercise in software engineering and algorithmic understanding, demystifying the "black box" of modern ML tools.

### 1.2 Core Features

*   **Custom Data Pipeline**:
    *   **Ingestion**: A custom downloader using `urllib` fetches datasets, while a scratch-built `csv` parser handles data ingestion without Pandas.
    *   **Feature Engineering**: A specialized preprocessing module groups data by location to capture the high variance in Dhaka's housing market (e.g., Gulshan vs. Dakshinkhan).
*   **Web Interface**:
    *   **Backend**: A lightweight HTTP server built using Python's `http.server` module handles API requests.
    *   **Frontend**: A responsive UI built with vanilla HTML, JavaScript, and Tailwind CSS (via CDN), ensuring no frontend framework dependencies.

### 1.3 The Algorithm: Linear Regression with Ridge Regularization

The predictive core is a **Linear Regression model with Ridge (L2) Regularization**. This variant was chosen to mitigate overfitting caused by high-dimensional feature spaces resulting from location encoding.

The model solves for the weight vector $\theta$ using the normal equation:

$$ \theta = (X^T X + \lambda I)^{-1} X^T Y $$

Where:
*   $X$ is the feature matrix ($n \times m$).
*   $Y$ is the target vector ($n \times 1$).
*   $\lambda$ is the regularization parameter.
*   $I$ is the identity matrix ($m \times m$).

**Implementation Details**:
Instead of using `numpy.linalg.inv`, the system implements **Gauss-Jordan Elimination** to invert the matrix $(X^T X + \lambda I)$. This involves:
1.  Augmenting the matrix with the identity matrix.
2.  Performing row operations to reduce the original matrix to the identity form.
3.  Extracting the inverse from the augmented section.

### 1.4 Why From Scratch?

Building from scratch is not about reinventing the wheel, but about understanding how the wheel rolls. By manually managing memory, handling numerical stability, and implementing $O(n^3)$ algorithms, this project exposes the engineering challenges that libraries like Scikit-learn abstract away. It proves that core computer science principles are more fundamental than tool proficiency.



## 2: The Experimentation

### 2.1 Objective

The secondary goal of this project was to benchmark the custom "from-scratch" implementation against Scikit-learn, the industry standard. To explore the limits of performance, the core mathematical engine was implemented in three distinct versions.

### 2.2 Implementations Analyzed

#### A. Pure Python (`model_impls/py_impl`)
*   **Approach**: Direct implementation of matrix operations using nested lists.
*   **Algorithm**: Naive matrix multiplication ($O(n^3)$) and Gauss-Jordan inversion.
*   **Characteristics**: High interpretability but suffers from Python's interpreter overhead in tight loops.

#### B. Single-Threaded C (`model_impls/c_impl`)
*   **Approach**: A C extension accessed via Python bindings.
*   **Memory Management**: Manual allocation (`malloc`/`free`) for matrices.
*   **Algorithm**: Direct port of the Python logic to C.
*   **Goal**: To measure the speedup gained solely by moving from an interpreted language to a compiled one, without algorithmic changes.

#### C. Multi-Threaded C (`model_impls/c_pthread`)
*   **Approach**: High-performance C implementation using **POSIX Threads (pthreads)**.
*   **Parallelization**:
    *   **Row-wise Decomposition**: Matrix operations (multiplication, inversion) are split into chunks of rows, processed in parallel by a thread pool.
    *   **Dynamic Pooling**: The thread pool size is dynamically determined by `sysconf(_SC_NPROCESSORS_ONLN)`.
*   **Cache Optimization**:
    *   **Block Matrix Multiplication (Tiling)**: Implements blocked multiplication with `BLOCK_SIZE=64` to maximize CPU cache hits (L1/L2) and reduce memory access latency.

### 2.3 Benchmark Results

Experiments were conducted on a Ryzen 5600G CPU using 5-fold cross-validation.

#### 1. Small Datasets (5,332 samples)
*   **Pure Python**: **~0.137s**
*   **Scikit-learn**: **~0.289s**
*   **Result**: Python is **~1.9x faster**.
*   **Analysis**: Scikit-learn incurs significant "warmup" overhead (loading NumPy, BLAS libraries) which dominates execution time for small tasks. The lightweight Python implementation starts instantly.

![Python Implementation Benchmark](images/graphs/speedtest_plots/py_impl_5fold_5332d_6f.png)

#### 2. Large Datasets (Scaling to 100k+)
*   **C (pthreads)**: **~0.01s**
*   **Scikit-learn**: **~0.02s**
*   **Result**: Multi-threaded C is **~2x faster**.
*   **Analysis**: On larger datasets, the algorithmic efficiency of Scikit-learn usually wins. However, the custom C implementation, optimized with threading and cache tiling, outperforms the generic BLAS routines used by Scikit-learn for this specific problem size.

![C Pthread Implementation Benchmark](images/graphs/speedtest_plots/c_pthread_5fold_5332d_6f.png)


_**Scripts used for benchmarks can be found at [benchmarks/scripts](./benchmarks/scripts/) directory.**_

### 2.4 Key Findings

1.  **Overhead Matters**: For micro-services or serverless functions handling small requests, heavy libraries like Pandas/Scikit-learn can be slower than simple Python loops due to initialization costs.
2.  **The Power of C**: A naive C port provides a ~10x speedup over Python.
3.  **Algorithmic Optimization**: Simply writing in C isn't enough. Achieving state-of-the-art performance requires hardware-aware optimizations like **cache tiling** and **multi-threading**, as demonstrated by the `c_pthread` implementation beating Scikit-learn.



## 3: Limitations & Future Scope

### 3.1 Current Limitations

*   **Linearity Assumption**: The model relies on linear regression, which assumes a linear relationship between features (e.g., area, bedrooms) and rent. However, real-world housing markets often exhibit non-linear behaviors, particularly in luxury or outlier segments.
*   **Feature Scope**: The current prediction is based on a limited set of features. Critical factors such as building age, proximity to public transport (metro stations), and neighborhood security are not currently accounted for.
*   **Data Sparsity**: Certain locations have very few data points, leading to high variance in predictions for those specific areas.

### 3.2 Future Engineering Plans

*   **Non-Linear Models**: Implementing Polynomial Regression or Decision Trees (from scratch, of course) to capture complex market dynamics.
*   **Expanded Feature Engineering**: Incorporating geospatial data to calculate distances to key amenities.
*   **Cloud Deployment**: Dockerizing the application for deployment on cloud platforms like AWS or Heroku, moving beyond the local `http.server`.

## 4. Conclusion

This project stands as a testament to the value of "learning by implementation." By stripping away the layers of abstraction provided by modern libraries, I gained a granular understanding of the mathematical and computational realities of machine learning. The result is not just a functioning house rent predictor, but a high-performance, scratch-built system that challenges the assumption that "custom" means "slower."
