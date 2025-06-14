# Performance Analysis and Benchmarking Tests

Since the scratch implementation of L2 Regularized Linear Regression seems working, it is a must to benchmark the errors and performance of the custom implementation against the implementation of a popular library. For this purpose, I will analyze the scratch built model's performance against the `scikit-learn` library's `LinearRegression` class. The goal is to analyze and get an understanding about the scratch built model's data precision against popular libraries' implementation.

## Environment Setup

Before I proceed, I need to install scikit-learn library to perform the benchmark comparison tests. As I'm using uv, I added scikit-learn through uv by running the command:

```bash
uv add scikit-learn --group benchmark-test # I will surely not add as a standard dependency!
```

## Getting Benchmark Scripts Ready

I created two different benchmark scripts at [scripts/](./scripts/) directory. The first script is [01_benchmark_script.py](./scripts/01_benchmark_script.py) and the second script is [02_benchmark_script.py](./scripts/02_benchmark_script.py).  
The first script will run the benchmark without any modification of data preprocessing pipeline. The second script will run the benchmark with some modifications to the data preprocessing pipeline, to get a fair amount of data for running benchmark, although some locations **were skipped** because of not having enough data.  
The modification were done regarding `construct_location_from_area` function defined in [../app/handler/data/preprocess.py](../app/handler/data/preprocess.py) (line 36), which is slightly modified as `modified_construct_location_from_area` function defined in [02_benchmark_script.py](./scripts/02_benchmark_script.py) (line 26). The preprocessing pipeline is also changed, modification are made in [02_benchmark_script.py](./scripts/02_benchmark_script.py) (line 46).

## Running the Benchmark

I ran the benchmark scripts by running the commands:

```bash
python benchmarks/scripts/01_benchmark_script.py
python benchmarks/scripts/02_benchmark_script.py
```

## Documenting Results

Results are added in [results/](./results/) directory as in markdown files with corresponding names `01_benchmark_results.md` and `02_benchmark_results.md`.

## Creating Bar Plot of Results

I created a bar plot of the results using `matplotlib` library. The bar plot is added in [graphs/](./graphs/) directory as in directories with corresponding names `01_benchmark_results_bar_plot` and `02_benchmark_results_bar_plot`.

The scripts are added into [scripts/](./scripts/) directory, which are [01_create_bar_plot.py](./scripts/01_create_bar_plot.py) and [02_create_bar_plot.py](./scripts/02_create_bar_plot.py).
