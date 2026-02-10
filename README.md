# diabetes_prediction_system
This project is an AI-powered diagnostic tool that uses a Random Forest Classifier trained on over 700,000 healthcare records to predict the probability of diabetes based on patient demographics, lifestyle, and clinical vitals.

## Clone the Repository
Link : https://github.com/PesandiM/diabetes_prediction_system.git

## Download the Trained Model
Due to GitHub's file size limitations (100MB), the finalized Random Forest model (146MB) is hosted on Google Drive. <br>
Download link : https://drive.google.com/file/d/1tO-n0dxMvPeVdaAZWxpOQ9zgfXwxbcnO/view?usp=sharing <br>
Instructions: Create a folder named model in the project root (if it doesn't exist).<br> & Place the downloaded .pkl file inside the model/ folder.

## Install Dependencies
Install the required libraries using pip

## Run the application
edit configurations

<img width="995" height="704" alt="image" src="https://github.com/user-attachments/assets/a12d4bef-2aa8-427c-9415-f2264e665f78" />


## system architecture
The application follows a modular design to ensure scalability and ease of maintenance: <br>
1. app.py: The Streamlit frontend and UI logic. <br>
2. preprocessing.py: Handles data transformation, categorical encoding, and scaling to ensure user input matches the training data format. <br>
3. style.css: Custom styling for a clean, medical-grade user interface. <br>
4. model/: Contains the serialized Random Forest model.

## Model performance
Algorithm: Random Forest <br>
Recall (Sensitivity): 91.14% (Optimized for clinical screening)<br>
Accuracy: 71.68%
<br>
<br>
<br>
