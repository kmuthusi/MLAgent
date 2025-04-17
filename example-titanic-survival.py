# Example: Using MLAgent Suite to fit classification models to Titanic dataset
# uncomment to run this to install required packages
# check-and-install-packages.py

from mlagents import MLAgentClassifier

# read the data
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

# perform data manipulation
df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

# speficify features
X = df.drop(columns=['Survived'])

# specicify target
y = df['Survived']

# load agent
agent = MLAgentClassifier()

# apply agent to data
agent.load_data(X, y)
results = agent.train_and_evaluate(save_path="models\best_binary_classifier")
predictions = agent.predict()

# Example of loading the saved model
best_model = agent.load_best_model()
