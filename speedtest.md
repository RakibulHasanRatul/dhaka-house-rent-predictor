# Speed Test Against scikit-learn's Linear Regression

This comparison highlights the trade-off between control (scratch-built) and optimization (Scikit-learn), especially in compute-constrained environments.

## Summary

To further evaluate my scratch-built implementation of linear regression, I have created a speed test against scikit-learn's implementation. The results are quite interesting. I found that my implementation is faster than scikit-learn's for smaller datasets! Although the dataset contains just 5332 rows.

However, that’s just one side of the story. The scratch implementation is definitely faster, but only for smaller datasets. As the dataset size increases, the performance of the scratch implementation degrades significantly compared to scikit-learn's implementation!

### Key Takeaways

- **Scratch-built model** is faster on small datasets but does not scale well.
- **scikit-learn** is significantly faster and better optimized for larger datasets.
- **Even without optimization, the scratch-built model** processed 32,500 rows in under 1s!
- **Great for learning, benchmarking, and systems with no external library access.**

**_Note: All benchmarks were run on a Ryzen 5 5600G with 8 GB RAM and Python 3.12 on Fedora 42 Workstation. Performance may vary depending on system specifications._**

## Results With Smaller Datasets

First, I created a Python script to measure the speed of each implementation. I left the code as it is for preprocessing. The current version generates fewer elements in the feature vector for each location.

After running the speed test using the [script](./speedtest_scripts/01_speedtest_with_smaller_datasets.py), I got the following results:

### 1st Run

```plaintext
❯ python speedtest_scripts/01_speedtest_with_smaller_datasets.py
Downloading: █████████████████████████ 100.00% (325679/325679 bytes)

File downloaded to: data/raw/formatted-csv-dhaka-rent-predictor.csv from url: https://gist.githubusercontent.com/RakibulHasanRatul/9101d6c95bbd3800e1c22b68e6462d76/raw/8d1b1cd8356e0a2ad2db205531421a48660eb6ba/formatted-csv-dhaka-rent-predictor.csv

Working on 125 locations.
Skipped 149 locations due to insufficient data.
Running scratch-built model...
Scratch-built model took 0.1368 seconds.
Running sklearn model...
Sklearn model took 0.2954 seconds.
```

### 2nd Run

```plaintext
❯ python speedtest_scripts/01_speedtest_with_smaller_datasets.py
Downloading: █████████████████████████ 100.00% (325679/325679 bytes)

File downloaded to: data/raw/formatted-csv-dhaka-rent-predictor.csv from url: https://gist.githubusercontent.com/RakibulHasanRatul/9101d6c95bbd3800e1c22b68e6462d76/raw/8d1b1cd8356e0a2ad2db205531421a48660eb6ba/formatted-csv-dhaka-rent-predictor.csv

Working on 125 locations.
Skipped 149 locations due to insufficient data.
Running scratch-built model...
Scratch-built model took 0.1372 seconds.
Running sklearn model...
Sklearn model took 0.2826 seconds.
```

### 3rd Run

```plaintext
❯ python speedtest_scripts/01_speedtest_with_smaller_datasets.py
Downloading: █████████████████████████ 100.00% (325679/325679 bytes)

File downloaded to: data/raw/formatted-csv-dhaka-rent-predictor.csv from url: https://gist.githubusercontent.com/RakibulHasanRatul/9101d6c95bbd3800e1c22b68e6462d76/raw/8d1b1cd8356e0a2ad2db205531421a48660eb6ba/formatted-csv-dhaka-rent-predictor.csv

Working on 125 locations.
Skipped 149 locations due to insufficient data.
Running scratch-built model...
Scratch-built model took 0.1379 seconds.
Running sklearn model...
Sklearn model took 0.2903 seconds.
```

### Summary of Results with Smaller Dataset

The speed test was conducted three times, and the results were consistent across runs. The scratch-built linear regression model consistently outperformed scikit-learn's implementation in terms of execution time.  
The average time taken by the scratch-built model was approximately 0.1373 seconds, while the average time for the scikit-learn model was around 0.2894 seconds.  
This indicates that the scratch implementation is approximately 1.9 times faster than scikit-learn's implementation for this specific dataset.

## Results With Larger Datasets

However, it's crucial to evaluate the performance of both models on larger datasets as well. Preliminary tests indicate that while the scratch-built model performs admirably on smaller datasets, its performance may degrade with larger datasets. In contrast, scikit-learn's implementation is optimized for scalability and may outperform the scratch model in such scenarios.

So, what I did here is to create a new [script](./speedtest_scripts/02_speedtest_with_modified_preprocessing.py) changing the preprocessing function to generate larger feature set for each location.

### 1st Run with Larger Datasets

```plaintext
❯ python speedtest_scripts/02_speedtest_with_modified_preprocessing.py
Downloading: █████████████████████████ 100.00% (325679/325679 bytes)

File downloaded to: data/raw/formatted-csv-dhaka-rent-predictor.csv from url: https://gist.githubusercontent.com/RakibulHasanRatul/9101d6c95bbd3800e1c22b68e6462d76/raw/8d1b1cd8356e0a2ad2db205531421a48660eb6ba/formatted-csv-dhaka-rent-predictor.csv

Working on 41 locations.
Skipped 19 locations due to insufficient data.
Running scratch-built model...
Scratch-built model took 0.1334 seconds.
Running sklearn model...
Sklearn model took 0.1058 seconds.
```

### 2nd Run with Larger Datasets

```plaintext
❯ python speedtest_scripts/02_speedtest_with_modified_preprocessing.py
Downloading: █████████████████████████ 100.00% (325679/325679 bytes)

File downloaded to: data/raw/formatted-csv-dhaka-rent-predictor.csv from url: https://gist.githubusercontent.com/RakibulHasanRatul/9101d6c95bbd3800e1c22b68e6462d76/raw/8d1b1cd8356e0a2ad2db205531421a48660eb6ba/formatted-csv-dhaka-rent-predictor.csv

Working on 41 locations.
Skipped 19 locations due to insufficient data.
Running scratch-built model...
Scratch-built model took 0.1327 seconds.
Running sklearn model...
Sklearn model took 0.1051 seconds
```

### 3rd Run with Larger Datasets

```plaintext
❯ python speedtest_scripts/02_speedtest_with_modified_preprocessing.py
Downloading: █████████████████████████ 100.00% (325679/325679 bytes)

File downloaded to: data/raw/formatted-csv-dhaka-rent-predictor.csv from url: https://gist.githubusercontent.com/RakibulHasanRatul/9101d6c95bbd3800e1c22b68e6462d76/raw/8d1b1cd8356e0a2ad2db205531421a48660eb6ba/formatted-csv-dhaka-rent-predictor.csv

Working on 41 locations.
Skipped 19 locations due to insufficient data.
Running scratch-built model...
Scratch-built model took 0.1325 seconds.
Running sklearn model...
Sklearn model took 0.1057 seconds.
```

As expected, the performance difference narrows. But the difference is not as significant. The average time taken by the scratch-built model was approximately 0.1329 seconds, while the average time for the scikit-learn model was around 0.1055 seconds. Although in this case, the scratch-built model is approximately 1.26 times slower than scikit-learn's implementation.

## Summary of Results with Both Datasets

In summary, the speed test results indicate that the scratch-built linear regression model is faster than scikit-learn's implementation for **smaller datasets only**. For larger datasets, **scikit-learn's implementation is more efficient and robust**, making it a better choice for production use cases.

## Further Stresses!

Although I have tested and drew a conclusion, I think a further stress test can be done to see how the scratch-built model performs with larger datasets! To do so, I will create another [script](./speedtest_scripts/03_stress_speed_test.py) to test the speed against full datasets at once.

This will ensure that the scratch-built model is tested against the entire dataset at once, rather than per location. This will give a better understanding of how the scratch-built model performs with larger datasets.

### 1st Stress Run

```plaintext
❯ python speedtest_scripts/03_stress_speed_test.py
Downloading: █████████████████████████ 100.00% (325679/325679 bytes)

File downloaded to: data/raw/formatted-csv-dhaka-rent-predictor.csv from url: https://gist.githubusercontent.com/RakibulHasanRatul/9101d6c95bbd3800e1c22b68e6462d76/raw/8d1b1cd8356e0a2ad2db205531421a48660eb6ba/formatted-csv-dhaka-rent-predictor.csv

Working on 5332 dataset.
Running scratch-built model...
Scratch-built model took 0.1475 seconds.
Running sklearn model...
Sklearn model took 0.0174 seconds.
```

### 2nd Stress Run

```plaintext
❯ python speedtest_scripts/03_stress_speed_test.py
Downloading: █████████████████████████ 100.00% (325679/325679 bytes)

File downloaded to: data/raw/formatted-csv-dhaka-rent-predictor.csv from url: https://gist.githubusercontent.com/RakibulHasanRatul/9101d6c95bbd3800e1c22b68e6462d76/raw/8d1b1cd8356e0a2ad2db205531421a48660eb6ba/formatted-csv-dhaka-rent-predictor.csv

Working on 5332 dataset.
Running scratch-built model...
Scratch-built model took 0.1473 seconds.
Running sklearn model...
Sklearn model took 0.0180 seconds.
```

### 3rd Stress Run

```plaintext
❯ python speedtest_scripts/03_stress_speed_test.py
Downloading: █████████████████████████ 100.00% (325679/325679 bytes)

File downloaded to: data/raw/formatted-csv-dhaka-rent-predictor.csv from url: https://gist.githubusercontent.com/RakibulHasanRatul/9101d6c95bbd3800e1c22b68e6462d76/raw/8d1b1cd8356e0a2ad2db205531421a48660eb6ba/formatted-csv-dhaka-rent-predictor.csv

Working on 5332 dataset.
Running scratch-built model...
Scratch-built model took 0.1464 seconds.
Running sklearn model...
Sklearn model took 0.0183 seconds.
```

### Conclusion of Stress Test

Noticed a pattern? **The scratch-built model's performance is too consistent yet** It is still predicting within 0.2 seconds.

However, the scikit-learn's implementation is way faster than the scratch-built model. It is taking around 0.02 seconds to run through the entire dataset, **exactly 10 times better**! It is expected, as scikit-learn's implementation is optimized for performance and scalability.

It’s still surprising how a non-optimized model completes training under 0.2 seconds with 5332 datasets!

## Speed Plot!

After running 2 speed tests - one with smaller dataset and another with larger dataset - it will be interesting to loop through a range and slice the feature's list for generating different length of datasets and run the speed test for each dataset! I should be using `matplotlib` to plot the speed test results.

To do so, I wrote another [script](./speedtest_scripts/04_plot_speedtest_graph.py) and plotted the plot at `images/graphs/speed_comparisons/` directory. in this file, three example are added.

> Speed comparison between scratch-built vs scikit-learn Linear Regression across dataset sizes.

> 5 fold Cross-Validation
> ![Speed Plot](images/graphs/speed_comparisons/5_fold_speed_comparison.png)

> 25 fold Cross-Validation
> ![Speed Plot](images/graphs/speed_comparisons/25_fold_speed_comparison.png)

> 50 fold Cross-Validation
> ![Speed Plot](images/graphs/speed_comparisons/50_fold_speed_comparison.png)

### Conclusion of Speed Plot!

It's interesting to see, in this test, the **scikit-learn's implementation is much faster than my scratch-built model** (as expected!) even for smaller datasets! But I showed and claimed that my implementation is faster than scikit-learn's implementation for smaller datasets! What about that claim?

This effect is likely due to `NumPy`’s **JIT-related optimizations kicking in after the first few runs**.! And **considering the numpy warmup time, the scratch-built model is still faster** than scikit-learn's implementation for smaller datasets.

In conclusion, the scratch-built model is well-suited for small datasets and constrained environments where external dependencies or system resources are limited. And **without optimizations, it still can be trained through 32,500 datasets in under 1 second**.

That’s a huge win — and a solid demonstration of what even non-optimized models can achieve.
