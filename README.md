# 🚗 Russian Car Plates Price Prediction 🚗

This project predicts the prices of Russian car plates using machine learning models, specifically Neural Networks (NN) built with **Keras**. The model is trained on data such as car type, region, number of characters, and other influential factors to predict car plate prices.

## 🚀 Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/SergKhachikyan/Russian_Car_Plates_Price_Predict.git
    ```

2. **Navigate to the project directory:**
    ```bash
    cd Russian_Car_Plates_Price_Predict
    ```

3. **Install the dependencies:**
    Ensure Python 3.x is installed, then run:
    ```bash
    pip install -r requirements.txt
    ```

4. **Launch the project:**
    Open the Jupyter Notebook interface:
    ```bash
    jupyter notebook
    ```

## 🔧 Technologies

This project uses the following technologies:
- **Python** 🐍: Programming language.
- **Keras** 🧠: Deep learning framework for building neural networks.
- **TensorFlow** 🔥: Backend for Keras to execute machine learning models.
- **Pandas** 📊: Data handling and analysis library.
- **Scikit-learn** 📚: Traditional machine learning models for comparison.
- **Matplotlib & Seaborn** 📈: Data visualization libraries.

## 📝 How to Use

1. **Prepare the dataset:**
    - Place your dataset in the `data/` folder. The dataset should contain information such as car type, region, number of characters, and other related features that influence the car plate prices.
  
2. **Run the analysis and train the model:**
    - Open Jupyter Notebook in the root directory:
    ```bash
    jupyter notebook
    ```
    - Open the relevant notebook (e.g., `car_plate_price_prediction.ipynb`) and execute the code cells sequentially to train the model.

3. **Make predictions:**
    After training the neural network, you can use the trained model to predict the price for new car plates. The function `predict_price()` can be used to input features and get the predicted price.

## 💡 Features

- **Data Exploration** 📊: Explore how various features like car type, region, and number of characters in the plate affect the price.
- **Neural Network Model** 🔮: Build and train a neural network using **Keras** to predict car plate prices.
- **Data Visualization** 🌈: Visualize how different features affect the price.
- **Model Evaluation** 📉: Evaluate model performance using metrics like Mean Squared Error (MSE).

## 🧠 Neural Network Architecture

- **Input Layer**: Accepts various features such as car type, region, and other factors.
- **Hidden Layers**: One or more hidden layers with activation functions (ReLU is commonly used).
- **Output Layer**: A single node that outputs the predicted price.
- **Optimization**: The model uses the Adam optimizer for efficient training.

Here’s a sample of how the neural network might be constructed in Keras:
python
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))  # First hidden layer
model.add(Dense(32, activation='relu'))  # Second hidden layer
model.add(Dense(1))  # Output layer

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=32)

🏆 Model Performance
 .Loss Function: The model uses Mean Squared Error (MSE) as the loss function, which is appropriate for regression tasks.

 .Metrics: The performance of the model is evaluated using MSE and other regression metrics.

📊 Visualizations
 .Price Distribution: Visualize the distribution of car plate prices across different regions and car types using histograms.

![download](https://github.com/user-attachments/assets/016dcb59-cfeb-42b7-bb70-67f9380677ab)


 .Correlation Heatmap: Visualize the correlations between different features to understand their relationships with the price.
