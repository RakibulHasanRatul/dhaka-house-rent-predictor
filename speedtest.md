# Speed Test Against Scikit-Learn's Linear Regression

Well, for further look to my scratch implementation of linear regression, I have created a speed test against Scikit-Learn's implementation. The results are quite interesting. I found that my implementation is faster than Scikit-Learn's for this dataset! Although this dataset has only 5332 rows only.

But, it's only the one side of the story. The scratch implementation is definitely faster, but only for smaller datasets. As the dataset size increases, the performance of the scratch implementation degrades significantly compared to Scikit-Learn's implementation!

## Results With Smaller Datasets

Firstly, I created a python script to measure the speeds of each implementations. I left the code as it is for preprocessing. The current version generates a less number of elements in the feature vector for each location.

After running the speed test using the [script](#script-i-used) provided below, I got the following results:

### 1st Run

```plaintext
❯ python speed_test.py
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
❯ python speed_test.py
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
❯ python speed_test.py
Downloading: █████████████████████████ 100.00% (325679/325679 bytes)

File downloaded to: data/raw/formatted-csv-dhaka-rent-predictor.csv from url: https://gist.githubusercontent.com/RakibulHasanRatul/9101d6c95bbd3800e1c22b68e6462d76/raw/8d1b1cd8356e0a2ad2db205531421a48660eb6ba/formatted-csv-dhaka-rent-predictor.csv

Working on 125 locations.
Skipped 149 locations due to insufficient data.
Running scratch-built model...
Scratch-built model took 0.1379 seconds.
Running sklearn model...
Sklearn model took 0.2903 seconds.
```

### Summary of Results

The speed test was conducted three times, and the results were consistent across runs. The scratch-built linear regression model consistently outperformed Scikit-Learn's implementation in terms of execution time.  
The average time taken by the scratch-built model was approximately 0.1373 seconds, while the average time for the Scikit-Learn model was around 0.2894 seconds.  
This indicates that the scratch implementation is approximately 1.9 times faster than Scikit-Learn's implementation for this specific dataset.  
This shows that for this specific dataset, the scratch implementation is faster than Scikit-Learn's implementation. It's important to note that this result may vary with different datasets and configurations, but it is a promising outcome for the scratch implementation.

## Results With Larger Datasets

However, it's crucial to evaluate the performance of both models on larger datasets as well. Preliminary tests indicate that while the scratch-built model performs admirably on smaller datasets, its performance may degrade with larger datasets. In contrast, Scikit-Learn's implementation is optimized for scalability and may outperform the scratch model in such scenarios.

So, what I did here is to run the same [script](#script-i-used) but with a small tweak. I changed the `construct_location_from_area` function in the `app/handler/data/preprocess.py`. It will ensure larger feature set is created for a secific location. The cnage I made is:

```python
def construct_location_from_area(addr: str) -> str:
    parts = [part.strip() for part in addr.split(",")]
    if len(parts) < 2:
        return addr.title()

    level2 = parts[-2]
    level1 = parts[-1]

    return f"{level2}, {level1}".title()
```

After changing the function, I ran the same script and got following results:

### 1st Run with Larger Datasets

```plaintext
❯ python speed_test.py
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
❯ python speed_test.py
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
❯ python speed_test.py
Downloading: █████████████████████████ 100.00% (325679/325679 bytes)

File downloaded to: data/raw/formatted-csv-dhaka-rent-predictor.csv from url: https://gist.githubusercontent.com/RakibulHasanRatul/9101d6c95bbd3800e1c22b68e6462d76/raw/8d1b1cd8356e0a2ad2db205531421a48660eb6ba/formatted-csv-dhaka-rent-predictor.csv

Working on 41 locations.
Skipped 19 locations due to insufficient data.
Running scratch-built model...
Scratch-built model took 0.1325 seconds.
Running sklearn model...
Sklearn model took 0.1057 seconds.
```

See, performance is degrading. The scratch-built model is still faster than Scikit-Learn's implementation, but the difference is not as significant as with both datasets. The average time taken by the scratch-built model was approximately 0.1329 seconds, while the average time for the Scikit-Learn model was around 0.1055 seconds. Although in this case, the scratch-built model is approximately 1.26 times faster than Scikit-Learn's implementation, the gap is narrowing as the dataset size increases.
This indicates that while the scratch implementation is still faster, it may not scale as well as Scikit-Learn's implementation for larger datasets. Further optimizations may be needed to improve the performance of the scratch-built model on larger datasets.

This speed test highlights the strengths and weaknesses of both implementations. The scratch-built model is faster for smaller datasets, but Scikit-Learn's implementation is more robust and scalable for larger datasets.

## Summary of Overall Results

In summary, the speed test results indicate that the scratch-built linear regression model is faster than Scikit-Learn's implementation for smaller datasets, but its performance degrades with larger datasets.

The scratch-built model is only efficient when working with too small datasets yet having resource limitations. For larger datasets, Scikit-Learn's implementation is more efficient and robust, making it a better choice for production use cases.

## Further Stresses!

Although I have tested and drew a conclusion, I think a further stress test can be done to see how the scratch-built model performs with larger datasets! To do so, I will change the [script](#script-i-used) a bit. Just changing the `run_speedtest` function will be enough.

Code snippet:

```python
def run_speedtest():
    preprocessed_data = preprocess_loaded_data(
        load_csv_data(download_csv_from_gist(FORMATTED_CSV_GIST_URL))
    )

    work_on_location = list(preprocessed_data.keys())

    x_total: list[list[float]] = []
    y_total: list[float] = []
    for location in work_on_location:
        x_total = x_total + preprocessed_data[location].feature_vectors
        y_total = y_total + preprocessed_data[location].labels

    dataset_len = len(x_total)

    print("Working on {} dataset.".format(dataset_len))

    def run_scratch_built_model():
        k = 5
        fold_size = dataset_len // k

        for fold in range(k):
            start = fold * fold_size
            end = start + fold_size

            x_test = x_total[start:end]
            y_test = y_total[start:end]

            x_train = x_total[:start] + x_total[end:]
            y_train = y_total[:start] + y_total[end:]

            weights = get_weight_vector(x_train, y_train)

            y_pred: list[float] = []
            for x in x_test:
                pred = sum(w[0] * xi for w, xi in zip(weights, x))
                y_pred.append(pred)

            r_squared(y_pred, y_test)
            mse(y_pred, y_test)
            mae(y_pred, y_test)

    def run_sklearn_model():
        k = 5
        fold_size = dataset_len // k

        for fold in range(k):
            start = fold * fold_size
            end = start + fold_size

            x_test = x_total[start:end]
            y_test = y_total[start:end]

            x_train = x_total[:start] + x_total[end:]
            y_train = y_total[:start] + y_total[end:]

            model = LinearRegression()
            model.fit(x_train, y_train)

            y_pred: list[float] = model.predict(x_test).tolist()

            r_squared(y_pred, y_test)
            mse(y_pred, y_test)
            mae(y_pred, y_test)

    print("Running scratch-built model...")
    scratch_time = timeit.timeit(run_scratch_built_model, number=1)
    print(f"Scratch-built model took {scratch_time:.4f} seconds.")
    print("Running sklearn model...")
    sklearn_time = timeit.timeit(run_sklearn_model, number=1)
    print(f"Sklearn model took {sklearn_time:.4f} seconds.")

```

This will ensure that the scratch-built model is tested against the entire dataset at once, rather than per location. This will give a better understanding of how the scratch-built model performs with larger datasets.

### 1st Stress Run

```plaintext
❯ python speed_test.py
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
❯ python speed_test.py
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
❯ python speed_test.py
Downloading: █████████████████████████ 100.00% (325679/325679 bytes)

File downloaded to: data/raw/formatted-csv-dhaka-rent-predictor.csv from url: https://gist.githubusercontent.com/RakibulHasanRatul/9101d6c95bbd3800e1c22b68e6462d76/raw/8d1b1cd8356e0a2ad2db205531421a48660eb6ba/formatted-csv-dhaka-rent-predictor.csv

Working on 5332 dataset.
Running scratch-built model...
Scratch-built model took 0.1464 seconds.
Running sklearn model...
Sklearn model took 0.0183 seconds.
```

### Conclusion of Stress Test

Noticed a pattern? **The scratch-built model is too consistent!** It is giving the output within 0.2 seconds! That's incredible!

But, the Scikit-Learn's implementation is way faster than the scratch-built model. It is taking around 0.02 seconds to run the entire dataset! It is expected, as Scikit-Learn's implementation is optimized for performance and scalability.

But shocked to see how a non optimized model performs so well with 5332 datasets!

## Script I Used

The python script I used for the speed test is as follows:

```python
from sklearn.linear_model import LinearRegression
import timeit
from app.handler.data.download import download_csv_from_gist
from app.handler.data.load import load_csv_data
from app.handler.data.preprocess import preprocess_loaded_data
from app.model.linear_regression import get_weight_vector
from config import FORMATTED_CSV_GIST_URL


def r_squared(y_predicted: list[float], y_original: list[float]) -> float:
    if len(y_original) != len(y_predicted):
        raise ValueError("Length of predicted and original lists must be the same.")

    y_mean = sum(y_original) / len(y_original)

    ss_total = sum((y - y_mean) ** 2 for y in y_original)
    if ss_total == 0:
        # All y values are (almost) the same, R squared value is undefined
        # treating as 0 for safe reporting
        return 0.0

    ss_residual = sum(
        (y_o - float(y_p)) ** 2 for y_o, y_p in zip(y_original, y_predicted)
    )

    return 1 - (ss_residual / ss_total)


def mse(y_predicted: list[float], y_original: list[float]) -> float:
    return sum(
        (float(y) - float(y_hat)) ** 2 for y, y_hat in zip(y_original, y_predicted)
    ) / len(y_original)


def mae(y_predicted: list[float], y_original: list[float]) -> float:
    return sum(
        abs(float(y) - float(y_hat)) for y, y_hat in zip(y_original, y_predicted)
    ) / len(y_original)


def run_speedtest():
    preprocessed_data = preprocess_loaded_data(
        load_csv_data(download_csv_from_gist(FORMATTED_CSV_GIST_URL))
    )

    locations = list(preprocessed_data.keys())

    work_on_location = [
        location
        for location in locations
        if len(preprocessed_data[location].feature_vectors) >= 10
    ]

    print("Working on {} locations.".format(len(work_on_location)))
    print(
        "Skipped {} locations due to insufficient data.".format(
            len(locations) - len(work_on_location)
        )
    )

    def run_scratch_built_model():
        for location in work_on_location:
            data = preprocessed_data[location]

            x_total = data.feature_vectors
            y_total = data.labels
            total = len(x_total)

            k = 5
            fold_size = total // k

            for fold in range(k):
                start = fold * fold_size
                end = start + fold_size

                x_test = x_total[start:end]
                y_test = y_total[start:end]

                x_train = x_total[:start] + x_total[end:]
                y_train = y_total[:start] + y_total[end:]

                weights = get_weight_vector(x_train, y_train)

                y_pred: list[float] = []
                for x in x_test:
                    pred = sum(w[0] * xi for w, xi in zip(weights, x))
                    y_pred.append(pred)

                r_squared(y_pred, y_test)
                mse(y_pred, y_test)
                mae(y_pred, y_test)

    def run_sklearn_model():
        for location in work_on_location:
            data = preprocessed_data[location]

            x_total = data.feature_vectors
            y_total = data.labels
            total = len(x_total)

            k = 5
            fold_size = total // k

            for fold in range(k):
                start = fold * fold_size
                end = start + fold_size

                x_test = x_total[start:end]
                y_test = y_total[start:end]

                x_train = x_total[:start] + x_total[end:]
                y_train = y_total[:start] + y_total[end:]

                model = LinearRegression()
                model.fit(x_train, y_train)

                y_pred: list[float] = model.predict(x_test).tolist()

                r_squared(y_pred, y_test)
                mse(y_pred, y_test)
                mae(y_pred, y_test)

    print("Running scratch-built model...")
    scratch_time = timeit.timeit(run_scratch_built_model, number=1)
    print(f"Scratch-built model took {scratch_time:.4f} seconds.")
    print("Running sklearn model...")
    sklearn_time = timeit.timeit(run_sklearn_model, number=1)
    print(f"Sklearn model took {sklearn_time:.4f} seconds.")


if __name__ == "__main__":
    run_speedtest()
```
