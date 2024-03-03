import pandas as pd
from matplotlib import pyplot as plt

# Importing and cleaning dataset
gamedf = pd.read_csv(r"C:\Users\Zain\Downloads\GameRecommender-main\GameRecommender-main\vgsales.csv")
temp_gamedf = pd.read_csv(r"C:\Users\Zain\Downloads\GameRecommender-main\GameRecommender-main\vgsales.csv")

# Identify the string columns in your dataframe
string_columns = gamedf.select_dtypes(include=['object']).columns

from sklearn.preprocessing import LabelEncoder

# Initialize the LabelEncoder object
label_encoder = LabelEncoder()
print(gamedf.head())
# Perform label encoding on string_columns
for column in string_columns:
    temp_gamedf[column] = label_encoder.fit_transform(temp_gamedf[column])

# Print the modified dataframe
print(temp_gamedf.head())

from sklearn.preprocessing import MinMaxScaler

# Identify the numerical columns in your dataframe
numerical_columns = temp_gamedf.select_dtypes(include=['float64', 'int64', 'int32']).columns
numerical_columns=numerical_columns.drop('Rank')
# Initialize the MinMaxScaler object
scaler = MinMaxScaler()

# Perform normalization on numerical_columns
temp_gamedf[numerical_columns] = scaler.fit_transform(temp_gamedf[numerical_columns])

# Print the modified dataframe with label encoding and normalization
print(temp_gamedf.head())

from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd
import numpy as np

# Assuming X contains your features and y contains the target variable (Game Title)
X = temp_gamedf.drop('Name', axis=1).dropna()
y = temp_gamedf['Name'].dropna()
print(X.describe())
# Find the common indices between X and y
common_indices = X.index.intersection(y.index)

# Filter X and y using the common indices
X = X.loc[common_indices]
y = y.loc[common_indices]

# Drop rows containing NaN values
X = X.dropna()

# Initialize the SelectKBest feature selector with f_classif test
selector = SelectKBest(score_func=f_classif, k=5)

# Apply feature selection on the dataset
X_selected = selector.fit_transform(X, y)

# Get the selected feature indices
selected_feature_indices = selector.get_support(indices=True)

# Get the selected feature names
selected_feature_names = X.columns[selected_feature_indices]
print(selected_feature_names)
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
import numpy as np

def Similarity(n1, n2, temp_gamedf):
    # Select features for training the models
    features = ['Genre', 'Year', 'Publisher']  # Add other relevant features
    print('Checkpoint 1')
    # Prepare the training data
    X_train = temp_gamedf[features].dropna()
    y_train = temp_gamedf.loc[X_train.index, 'Rank']  # Assuming 'Rank' is your target variable
    print('Checkpoint 2')
    # Initialize the models
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    knn_model = KNeighborsRegressor(n_neighbors=5)
    lr_model = LinearRegression()
    print('Checkpoint 3')
    # Train the models
    rf_model.fit(X_train, y_train)
    knn_model.fit(X_train, y_train)
    lr_model.fit(X_train, y_train)
    print('Checkpoint 4')
    # Prepare the input data for prediction
    input_data = temp_gamedf.loc[[n1, n2], features].dropna()

    print('Checkpoint 5')
    # Handle missing values in the input data
    if input_data.isnull().values.any():
        print('checkpoint 5.5')
        return np.inf  # Return a high similarity for cases with missing values
    print('Checkpoint 6')
    # Predict the target variable (Rank) using each model
    rf_similarity = np.abs(rf_model.predict(input_data.iloc[0].values.reshape(1, -1)) - rf_model.predict(X_train))
    knn_similarity = np.abs(knn_model.predict(input_data.iloc[0].values.reshape(1, -1)) - knn_model.predict(X_train))
    lr_similarity = np.abs(lr_model.predict(input_data.iloc[0].values.reshape(1, -1)) - lr_model.predict(X_train))
    print('Checkpoint 7')
    # Select the highest similarity score among the models
    max_similarity = np.max(np.stack([rf_similarity, knn_similarity, lr_similarity]), axis=0)

    return max_similarity
import numpy as np

def getNeighbors(basegame, K, temp_gamedf):
    # Extract features for the base game
    basegame_features = basegame[['Genre', 'Year', 'Publisher']].values.reshape(1, -1)
    print('Neighbors checkpoint 1')
    # Extract features for all games
    all_game_features = temp_gamedf[['Genre', 'Year', 'Publisher']].values
    print('Neighbors checkpoint 2')
    # Handle missing or non-numeric values in features
    nan_mask = np.isnan(all_game_features.astype(float)).any(axis=1)
    all_game_features = all_game_features[~nan_mask]
    all_game_ranks = temp_gamedf.loc[~nan_mask, 'Rank']
    print('Neighbors checkpoint 3')
    # Calculate similarities using vectorized operations
    similarities = np.abs(Similarity(basegame_features, all_game_features, temp_gamedf))
    print('Neighbors checkpoint 4')
    # Sort and get the indices of the K nearest neighbors
    neighbor_indices = np.argsort(similarities)[:K]
    print('Neighbors checkpoint 5')
    # Extract the neighbor information
    neighbors = [(all_game_ranks.iloc[i], similarities[i]) for i in neighbor_indices]

    return neighbors
def predict_score(temp_gamedf):
    name = 'Call of Duty'
    matching_games = gamedf[gamedf['Name'].str.contains(name)]

    if matching_games.empty:
        print(f"No games found with a name containing '{name}'.")
        return

    new_game = matching_games.iloc[0].to_frame().T
    print('Selected Game:', new_game['Name'].values[0])

    K = 5
    neighbors = getNeighbors(new_game, K, temp_gamedf)
    print('\nRecommended Games:\n')

    for neighbor_info in neighbors:
        neighbor_rank, distance = neighbor_info
        neighbor_game = gamedf[gamedf['Rank'] == neighbor_rank]
        if not neighbor_game.empty:
            print(neighbor_game.iloc[0]['Name'])

# Call the predict_score function with the temp_gamedf DataFrame
predict_score(temp_gamedf)
