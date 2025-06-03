import pandas as pd  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler  
from sklearn.neighbors import KNeighborsRegressor  
from sklearn.metrics import mean_absolute_error, mean_squared_error  
from sklearn.model_selection import GridSearchCV  
import numpy as np  

np.random.seed(0)  
X = np.random.rand(100, 5)  
y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.randn(100)  
df = pd.DataFrame(X, columns=['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5'])  
df['Target'] = y  
X = df.drop(['Target'], axis=1)  
y = df['Target']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

scaler = StandardScaler()  
X_train_scaled = scaler.fit_transform(X_train)  
X_test_scaled = scaler.transform(X_test)  

model = KNeighborsRegressor()  
param_grid = {  
    'n_neighbors': [3, 5, 10, 15],  
    'weights': ['uniform', 'distance'],  
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']  
}  

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')  
grid_search.fit(X_train_scaled, y_train)  

best_model = grid_search.best_estimator_  
best_params = grid_search.best_params_  
y_pred = best_model.predict(X_test_scaled)  

print("Best Parameters: ", best_params)  
print("Mean Absolute Error: ", mean_absolute_error(y_test, y_pred))  
print("Root Mean Squared Error: ", np.sqrt(mean_squared_error(y_test, y_pred)))
