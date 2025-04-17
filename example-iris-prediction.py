# Example: Using MLAgent Suite to fit models to Iris dataset
# uncomment to run this to install required packages
# check-and-install-packages.py

from mlagents import MLAgentClassifier

# load Iris dataset
from sklearn.datasets import load_iris
iris = load_iris()

# specify features
X = pd.DataFrame(iris.data, columns=iris.feature_names)
feature_names = iris.feature_names

# specify target
y = iris.target

# load MLAgentClassifier
agent = MLAgentClassifier()

# apply MLAgent to Iris data
agent.load_data(X, y, feature_names=feature_names)
agent.train_and_evaluate(save_path = "models\best_models")
predictions = agent.predict()

# Example of loading the saved model
best_model = agent.load_best_model()
