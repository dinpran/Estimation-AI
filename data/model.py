import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import ssl
import urllib.request
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler,minmax_scale
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class CO2EmissionModel:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CO2EmissionModel, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        # Disable SSL certificate verification
        ssl._create_default_https_context = ssl._create_unverified_context

        # Load the data
        url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv'
        self.df = pd.read_csv(url)
        msk = np.random.rand(len(self.df)) < 0.8
        train = self.df[msk]
        test = self.df[~msk]

        # Train the model
        self.regr = LinearRegression()
        x_train = np.asanyarray(train[['ENGINESIZE', 'CYLINDERS']])
        y_train = np.asanyarray(train[['CO2EMISSIONS']])
        self.regr.fit(x_train, y_train)

    def predict(self, engine_size, cylinders):
        x_test = np.array([[engine_size, cylinders]])
        predicted_emission = self.regr.predict(x_test)
        return predicted_emission[0][0]

class BreastCancerModel:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BreastCancerModel, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        # Disable SSL certificate verification
        ssl._create_default_https_context = ssl._create_unverified_context

        # Load the data from local file
        url = 'data/can.csv'  # Adjust the file path as per your directory structure

        # Read the CSV file
        self.df = pd.read_csv(url)

        # Ensure dataset contains samples from both classes
        msk = np.random.rand(len(self.df)) < 0.8
        train = self.df[msk]
        test = self.df[~msk]

        # Train the logistic regression model
        self.log_reg = LogisticRegression(solver='liblinear')  # Set solver parameter
        x_train = np.asanyarray(train[['radius_mean', 'texture_mean','perimeter_mean','area_mean']])
        y_train = np.asanyarray(train[['diagnosis']])
        self.log_reg.fit(x_train,y_train)

    def predict(self, radius_mean, texture_mean, perimeter_mean, area_mean):
        features = np.array([[radius_mean, texture_mean, perimeter_mean, area_mean]])
        predicted_diagnosis = self.log_reg.predict(features)
        return predicted_diagnosis[0][0]

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import ssl

class HousePricePredictionModel:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(HousePricePredictionModel, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        # Disable SSL certificate verification
        ssl._create_default_https_context = ssl._create_unverified_context

        # Load the data
        url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/real_estate_data.csv'
        self.df = pd.read_csv(url)
        X = self.df.drop(columns=["MEDV"])
        y = self.df["MEDV"]

        # Split dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

        # Train the model
        self.regression_tree = DecisionTreeRegressor(criterion='squared_error')
        self.regression_tree.fit(X_train, y_train)

    def predict(self, crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, lstat):
        # Create a 2D array with the input features
        x_test = np.array([[crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, lstat]])
        # Predict the house price
        predicted_price = self.regression_tree.predict(x_test)
        return predicted_price[0]
    
class ChurnPredictionModel:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ChurnPredictionModel, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        # Load the data
        ssl._create_default_https_context = ssl._create_unverified_context
        url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv"
        churn_df = pd.read_csv(url)
        churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'churn']]
        churn_df['churn'] = churn_df['churn'].astype('int')

        X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
        y = np.asarray(churn_df['churn'])

        self.scaler = StandardScaler().fit(X)
        X = self.scaler.transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

        self.model = LogisticRegression(C=0.01, solver='liblinear').fit(X_train, y_train)

    def predict(self, tenure, age, address, income, ed, employ, equip):
        input_features = np.array([[tenure, age, address, income, ed, employ, equip]])
        input_features = self.scaler.transform(input_features)
        prediction = self.model.predict(input_features)
        return prediction[0]
    
class FuelConsumptionModel:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FuelConsumptionModel, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        # Disable SSL certificate verification
        ssl._create_default_https_context = ssl._create_unverified_context

        # Load the data
        url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv'
        self.df = pd.read_csv(url)
        msk = np.random.rand(len(self.df)) < 0.8
        train = self.df[msk]
        test = self.df[~msk]

        # Train the model
        self.regr = LinearRegression()
        x_train = np.asanyarray(train[['ENGINESIZE', 'CYLINDERS','CO2EMISSIONS']])
        y_train = np.asanyarray(train[['FUELCONSUMPTION_COMB']])
        self.regr.fit(x_train, y_train)

    def predict(self, engine_size, cylinders,co2_emission):
        x_test = np.array([[engine_size, cylinders,co2_emission]])
        predicted_emission = self.regr.predict(x_test)
        return predicted_emission[0][0]
    
class YachtModel:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(YachtModel, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        # Load data
        data = pd.read_csv(
            "data/yacht_hydrodynamics.data",
            sep=" ",
            names=[
                "Long Position", 
                "Prismatic coefficient", 
                "length-displacement ratio", 
                "bean-draught ratio", 
                "length-bean ratio", 
                "froude number", 
                "residuary resistance"
            ],
            on_bad_lines='skip'
        )
        data = data.fillna(0)

        # Convert data to tensors
        features = torch.tensor(data[[
            "Long Position", 
            "Prismatic coefficient", 
            "length-displacement ratio", 
            "bean-draught ratio", 
            "length-bean ratio", 
            "froude number"
        ]].values, dtype=torch.float32)
        
        target = minmax_scale(data[["residuary resistance"]], feature_range=(0, 1), axis=0, copy=True)
        target = torch.tensor(target, dtype=torch.float32)

        # Split data into training and testing sets
        train_features, test_features, train_target, test_target = train_test_split(
            features, target, test_size=0.2, random_state=100)

        # Define a custom dataset class
        class DatasetT(Dataset):
            def __init__(self, features, labels):
                self.X = features
                self.Y = labels

            def __len__(self):
                return len(self.X)

            def __getitem__(self, index):
                x = self.X[index]
                y = self.Y[index]
                return x, y

        # Create data loaders
        trainset = DatasetT(train_features, train_target)
        testset = DatasetT(test_features, test_target)
        train_loader = DataLoader(dataset=trainset, batch_size=277, shuffle=True)
        test_loader = DataLoader(dataset=testset, batch_size=62, shuffle=True)

        # Define the neural network model
        class Network(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(6, 12)
                self.fc2 = nn.Linear(12, 1)

            def forward(self, x):
                x = torch.sigmoid(self.fc1(x))
                x = self.fc2(x)
                return x

        # Initialize the model
        self.model = Network()

        # Define the loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optm = optim.Adam(self.model.parameters(), lr=0.0003)

        # Train the model
        for epoch in range(500):
            for x, y in train_loader:
                self.optm.zero_grad()                   
                outputs = self.model(x)                        
                loss = self.criterion(outputs, y)            
                loss.backward()                         
                self.optm.step()

    def predict(self, long_position, prismatic_coefficient, length_displacement_ratio, bean_draught_ratio, length_bean_ratio, froude_number):
        input_values = torch.tensor([[long_position, prismatic_coefficient, length_displacement_ratio, bean_draught_ratio, length_bean_ratio, froude_number]])
        output = self.model(input_values.float())
        return output.detach().numpy()[0][0]




