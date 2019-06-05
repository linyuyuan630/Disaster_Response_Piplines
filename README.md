# Disaster Response Pipeline Project

## Intriduction
This is a Udacity project that apply data engineering skills to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.

## Installation

- Python 3
- Anaconda distribution to install Python, since the distribution includes all necessary Python libraries as well as Jupyter Notebooks.
- Numpy
- Pandas
- Matplotlib
- Seaborn
- sklearn
- sqlalchemy
- nltk
- pickle

## File Structure
- app

   - template
      - master.html  # main page of web app
      - go.html  # classification result page of web app
   - run.py  # Flask file that runs app


- data

   - disaster_categories.csv  # data to process 
   - disaster_messages.csv  # data to process
   - process_data.py # the python file taht conducts the Extract, Transform, and Load process
   - InsertDatabaseName.db   # database to save clean data to a pickle file

- models

   - train_classifier.py # the python file that runs machine learning pipeline and saved the trained model as 
   - classifier.pkl  # saved model 
   

- README.md


## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

   If the above link does not work, Try the following steps:
   Run the app with "python run.py" command
   Open another terminal and type env|grep WORK this will give you the spaceid (it will start with view*** and some characters after that)
   Now open the browser window and type https://viewa7a4999b-3001.udacity-student-workspaces.com, replace the whole viewa7a4999b with the
   space id that you got after running env|grep WORK
   Press enter and the app should now run for you
   
 ## Acknowledgement
 Data was originally from Figure Eight. Lots of the codes are provided by Udacity.
