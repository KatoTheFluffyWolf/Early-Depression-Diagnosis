#!pip install kagglehub[pandas-datasets] -q
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import kagglehub
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense
from tensorflow.keras.models import Model
from kagglehub import KaggleDatasetAdapter
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import plot_model

# Set the path to the file you'd like to load
file_path = "student_depression_dataset.csv"

# Load the latest version
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "adilshamim8/student-depression-dataset",
  file_path,
  # Provide any additional arguments like
  # sql_query or pandas_kwargs. See the
  # documenation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)
# Cleaning the dataset, which involves removing irrelevant values in the 'city' column
irrelevant_cities = ['M.Tech', '3.0', "'Less than 5 Kalyan'", 'ME', 'M.Com', "'Less Delhi'", 'City']
df = df[~df['City'].isin(irrelevant_cities)].copy()
df = df.reset_index(drop=True)

#Encoding categorical features
numerical_features = [
    "CGPA", "Academic Pressure", "Study Satisfaction", "Work/Study Hours", "Financial Stress"
]
#Normalizing numerical feature
df[numerical_features] = df[numerical_features].replace('?', np.nan)
df = df.dropna(subset=numerical_features)
df[numerical_features] = df[numerical_features].astype(float)
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

#Numerical input
numerical_input = Input(shape=(len(numerical_features),), name='numerical_input')


#Encoding categorical features
def LabelEncoding(df,column):
  le = LabelEncoder()
  df[column] = le.fit_transform(df[column])
  return df

Categorical_features = {
    "Gender": df["Gender"].nunique(),
    "City": df["City"].nunique(),
    "Sleep Duration": df["Sleep Duration"].nunique(),
    "Dietary Habits": df["Dietary Habits"].nunique(),
    "Degree": df["Degree"].nunique(),
    "Have you ever had suicidal thoughts ?": df["Have you ever had suicidal thoughts ?"].nunique(),
    "Family History of Mental Illness": df["Family History of Mental Illness"].nunique()
}
for i in Categorical_features:
  df = LabelEncoding(df,i)


inputs = []
inputs.append(numerical_input)
#Add embedding layers to categorical inputs
embeddings = []


for feature_name in Categorical_features:
    input_layer = Input(shape=(1,), name=f'{feature_name}_input')
    embedding_layer = Embedding(input_dim=Categorical_features[feature_name] + 1, output_dim=4)(input_layer)
    flattened_layer = Flatten()(embedding_layer)

    inputs.append(input_layer)
    embeddings.append(flattened_layer)

merged_embeddings = Concatenate()(embeddings)
full_input = Concatenate()([merged_embeddings, numerical_input])

x = Dense(64, activation='relu')(full_input)
x = Dense(32, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)  # Binary classification (Depression yes/no)

# Build and compile model
model = Model(inputs=inputs, outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

y = df['Depression'].values

X_categorical = []
for feature in Categorical_features:
  X_categorical.append(df[feature].values.reshape(-1, 1))

# Numerical inputs
X_num = df[numerical_features].values

# Combine all inputs
X_all = [X_num] + X_categorical

X_train = X_all
y_train = y

early_stop = EarlyStopping(
    monitor='val_loss',      # Watch validation loss
    patience=3,              # If val_loss doesn't improve for 3 epochs, stop
    restore_best_weights=True  # Load the best weights (not last weights)
)

# --- Train the model ---
history = model.fit(
    X_train,
    y_train,
    batch_size=32,
    epochs=15,
    validation_split=0.2,  # 20% validation split from training data
    callbacks=[early_stop],
    verbose=1
)


plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

model.save("Depression_dectection.keras")