import warnings

warnings.filterwarnings("ignore")

from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def findKElbow(df, QI, show_plot=True):
    X = np.unique(df[QI])
    X = np.reshape(X, (-1, 1))
    kmeans = KMeans(n_init="auto").fit(X)
    visualizer = KElbowVisualizer(kmeans, k=(2, X.size))
    visualizer.fit(X)  # Fit the data to the visualizer
    if show_plot:
        visualizer.set_title(f"Elbow for QI {QI}")
        plt.show()
    else:
        plt.close()

    return visualizer.elbow_value_


def anonymizeSeparatiz(df: pd.DataFrame, QIs: list, k: int, type: str = "float", elbow_iterations: int = 10):
    def findMean(arr, n, m, type):
        if type == "int":
            return np.round(np.mean(arr[n : m + 1]))
        elif type == "float":
            return np.mean(arr[n : m + 1])

    def replaceMean(arr, n, m, mean):
        for i in range(n, m + 1):
            arr[i] = mean
        return arr

    new_df = df.copy()

    ncp = 0

    for QI in QIs:

        _k = k

        if k == -1:
            k_values = []
            for _ in range(elbow_iterations):
                k_values.append(findKElbow(new_df, QI, False))
            # print(k_values)
            _k = int(np.array(k_values).mean())

        print(f'k={_k}, QI={QI}')

        new_df = new_df.sort_values(QI)
        X = new_df[QI]
        X = np.reshape(X, (-1, 1))

        Ui = X[-1][0] # max
        Li = X[0][0]  # min

        n = 0
        for i in range(1, _k+1):
            sep = np.percentile(X, ((100 / _k) * i), axis=0, keepdims=False, method="closest_observation")
            # print(f"sep={sep}")
            m = np.max(np.where(X == sep[0]))  # m is the separator id
            print(f"m={m}")

            Uij = X[m][0]
            Lij = X[n][0]

            ncp += ((Uij - Lij) * (m-n+1)) / (Ui - Li)

            mean = findMean(X, n, m, type)
            X = replaceMean(X, n, m, mean)
            n = m + 1

        new_df[QI] = X
        print()
    new_df.sort_index(inplace=True)


    n_registers = new_df.shape[0]
    n_attributes = new_df.shape[1]
    ncp = ncp / (n_registers * n_attributes)
    print(f'ncp={ncp}')

    return new_df


def main():
    # X = np.array((20,22,25,31,32,33,37,38,39))

    df = pd.read_csv("datasets/adult.csv")

    k_mean = []
    for i in range(0, 10):
        k_mean.append(findKElbow(df, "age"))
    k = int(np.round(np.mean(k_mean)))
    print("k_mean", k_mean)
    print("desvio", np.std(k_mean))
    print(anonymizeSeparatiz(df, ["age"], k))


if __name__ == "__main__":
    main()
