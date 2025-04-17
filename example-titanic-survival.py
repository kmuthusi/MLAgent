# Example: Using MLAgent Suite to fit classification models to Titanic dataset
# uncomment to run this to install required packages
# check-and-install-packages.py

from mlagents import MLAgentClassifier

# Read the data
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

# Perform data manipulation
df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

# Speficify features
X = df.drop(columns=['Survived'])

# Specicify target
y = df['Survived']

# Initialize agent
agent = MLAgentClassifier()

# Load data
agent.load_data(X, y)

# Train on select algorithms
results = agent.train_and_evaluate(model_to_train=['XGBoost', 'Keras MLP', 'Random Forest'], save_path="models\best_binary_classifier")

# Make predict
predictions = agent.predict()

# Example of loading the saved model
best_model = agent.load_best_model()
