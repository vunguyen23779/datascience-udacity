# Disaster Response Pipeline Project

## Instructions:
This project I have learned and built on your data engineering skills to expand your opportunities and potential as a data scientist.
I created a machine learning pipeline to categorize disaster events so that you can send the messages to an appropriate disaster relief agency.

File structure:
app    

| - template    
| |- master.html : Home page
| |- go.html : Model result displays in this page
|- run.py : Python source code to initialize Flask webapp 


data    

|- disaster_categories.csv : Raw dataset
|- disaster_messages.csv : Raw dataset
|- process_data.py : Data cleaning pipeline    
|- DisasterResponse.db : Database to save clean data     


models   

|- train_classifier.py # Machine learning pipeline     
|- classifier.pkl # Result model     


1. Run the following commands in the project's root directory to set up your database and model.

    - To create a processed sqlite db
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To train and save a pkl model
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
    - To deploy the application locally
        `python run.py`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage
