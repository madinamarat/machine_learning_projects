# Machine Learning projects

This repository contains Data science projects that I have completed during the first weeks at [Spiced Academy Data Science Bootcamp](https://www.spiced-academy.com/en). The projects cover data analysis and vizualisation, feature engineering, application of different Machine Learning models (Random Forest, Linear Regression, Decision Tree, Naive Bayes, ARIMA), hyperparameter optimzation and regularization, natural languge processing and time series analysis.

### Gapminder - Visual Data Analysis 
* Data: [Gapminder](https://www.gapminder.org/data/)
* Goal: creating and animated scatterplot
* Descriptive statistics, plotting with Matplotlib and Seaborn
![Alt Text](https://github.com/madinamarat/machine_learning_projects/blob/master/01_animated_scatterplot/output.gif)

### Titanic - Classification Project
* Data: [Titanic-Kaggle dataset](https://www.kaggle.com/c/titanic/data)
* Goal: applying a machine learning model to predict passenger survival on the Titanic
* Exploring the Titanic dataset and building a baseline model
* Training a Logistic Regression and Random Forest classification models
* Creating features using one-hot encoding
* Calculating the train and test accuracy and cross-validation score
![Alt Text](https://github.com/madinamarat/machine_learning_projects/blob/master/02_titanic/data/age_distribution.png)

### Rental bikes - Regression Models
* Data: [Bike Sharing Demand - Kaggle](https://www.kaggle.com/c/bike-sharing-demand/data)
* Goal: to predict the demand of rental bikes by using a regression model
* Exploratory data analysis with Matplotlib and Seaborn, exploring time series data
* Feature engineering and features selection
* Training a model Linear Regression, Random Forest Regressor
* Cross-validation
* Optimizing the model with grid search

![Alt text](https://github.com/madinamarat/machine_learning_projects/blob/master/03_bikes/data/rent_by_hours.png)

### Lyrics - Text Classification
* Goal: to build a text classification model to predict the artist from a piece of text
* Scraping data from the [Lyrics website](https://www.lyrics.com/)
* Downloading HTML pages
* Getting a list of song urls with Regex
* HTML parsing with BeautifulSoup 
* NLP: tokenizing and lemmatizing by using Spacy
* Converting text to numbers by applying TfidfVectorizer and Bag Of Words method
* Building and training Naive Bayes, Random Forest, Logstic Regression, Decision Tree classifiers

### Temperature prediction - Time Series Analysis
* Data: [European Climate Assessment Dataset](https://www.ecad.eu/)
* Goal: to create a short-term temperature forecast
* Time series analysis and plotting
* Modelling trend and seasonality components
* Decomposing and recomposing time series data
* Building a baseline model
* Exploring Linear Autoregression
* Applying ARIMA model to predict the daily mean temperature

