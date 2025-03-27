# Example: Using MLAgent to fit models to Titanic dataset
# uncomment to run this to install required packages
# check-and-install-packages.py

if __name__ == "__main__":
    # read the data
    df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

    # perform data manipulation
    df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

    # speficify features
    X = df.drop(columns=['Survived'])

    # specicify target
    y = df['Survived']

    # load agent
    agent = MLAgent()

    # apply agent to data
    agent.load_data(X, y)
    results = agent.train_and_evaluate()
    predictions = agent.predict()
