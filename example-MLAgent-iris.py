# Example: Using MLAgent to fit models to Iris dataset
if __name__ == "__main__":

    # load Iris dataset
    from sklearn.datasets import load_iris
    iris = load_iris()

    # specify features
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    feature_names = iris.feature_names

    # specify target
    y = iris.target

    # load MLAgent
    agent = MLAgent()

    # apply MLAgent to Iris data
    agent.load_data(X, y, feature_names=feature_names)
    agent.train_and_evaluate()
    predictions = agent.predict()