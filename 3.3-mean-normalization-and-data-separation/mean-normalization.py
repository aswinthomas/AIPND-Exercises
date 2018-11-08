import numpy as np

def main():
    # Create a 1000 x 20 ndarray with random integers in the half-open interval [0, 5001).
    X = np.linspace(0,5001,1000*20).reshape(1000,20)

    # print the shape of X
    print("Shape of X:{}".format(X.shape))

    # Average of the values in each column of X
    ave_cols = np.average(X, axis=0)

    # Standard Deviation of the values in each column of X
    std_cols = np.std(X, axis=0)

    # Print the shape of ave_cols
    print("Average column shape:{}".format(ave_cols.shape))

    # Print the shape of std_cols
    print("Stddev column shape:{}".format(std_cols.shape))

    # Mean normalize X
    X_norm = (X-ave_cols)/std_cols

    # Print the average of all the values of X_norm
    print("X norm average:{}".format(np.average(X_norm)))

    # Print the average of the minimum value in each column of X_norm
    print("X norm average of each column minimum:{}".format(np.average(np.min(X_norm, axis=0))))

    # Print the average of the maximum value in each column of X_norm
    print("X norm average of each column minimum:{}".format(np.average(np.max(X_norm, axis=0))))

    # Create a rank 1 ndarray that contains a random permutation of the row indices of `X_norm`
    row_indices = np.random.permutation(X_norm.shape[0])
    train_index = int(0.6*row_indices.shape[0])
    crossval_index = int(0.8*row_indices.shape[0])
    test_index = row_indices.shape[0]
    print("Indexes:",train_index,crossval_index,test_index)
    row_indices_split = np.split(row_indices, [train_index,crossval_index,test_index])

    # Create a Training Set 60%
    X_train = X_norm[row_indices_split[0]]

    # Create a Cross Validation Set 20%
    X_crossVal = X_norm[row_indices_split[1]]

    # Create a Test Set 20%
    X_test = X_norm[row_indices_split[2]]

    # Print the shape of X_train
    print("Shape of X_train", X_train.shape)

    # Print the shape of X_crossVal
    print("Shape of X_crossVal", X_crossVal.shape)

    # Print the shape of X_test
    print("Shape of X_test", X_test.shape)


if __name__ == '__main__':
    main()