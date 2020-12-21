## GDP Growth Forecast

### INTRODUCTION

Gross domestic product (GDP) is a monetary measure of the market value
of all the final goods and services produced in a specific period. GDP is
often used as a metric for international comparisons as well as a broad measure
of economic progress. It is often considered to be the "world’s most powerful
statistical indicator of national development and progress".
You are provided with a dataset containing over a thousand annual indicators of economic development (including GDP and GDP growth) in hundreds of
countries around the world from 1960 to 2010, and you are asked to develop a
model that can predict GDP growth using data from the past. You can use all the
variables that you want from previous years, and you can use the model that you
feel is going to work better. The aim is to have a model that can predict next
year's GDP growth for each country in the world using the available data.

This project has an SQL database with 6 tables:
- Countries: Descriptive information about the countries (e.g. economical group, yeat latest data, etc.).
- CountryNotes, Footnotes: Source of each indicator for each country.
- CountryIndicators: All the variables avaiable per country/year.
- Indicators: Variables and its definition.
- EstimatedGPDGrowth: where the prediction values are stored.

Size of the dataset: 
- Initially: 1330 variables, 248 zones, 52 years (1960-2010).
- 22 zones are filtered out and only countries (228) are kept.
- Missing values of NY.GDP.MKTP.KD.ZG (response variable) are not analyzed. The database has 7968 of complete cases (the total with missing values: 11527).


### FUNCTIONAL AND TECHNICAL DESCRIPTION OF THE SOFTWARE

The software predicts the GDP-Growth of all available countries into a dataset, for the year specified by the user. 

Our goal is to predict a panel of time series, one for each country. With this in mind, we selected a gradient tree-boosting model because they are known to have a high prediction accuracy. Moreover, our model has the characteristic that allows grouped data (here heach country is a group), which should improve the prediction performance.
One restriction of our project is that we must have at most 50 variables to do the predictions. That means we have to do feature selection before training the model. This can be divided in the following steps:

- Autoregressive model: We found that NY.GDP.MKTP.KD.ZG of one year can be explained by the prior year and we created the variable to improve the predicttion.
- As we know that the mentioned variable is important, we do a linear model (controlling by country) with it and extract the residuals, which will be used for the feature selection.
- Our dataset has many missing values. We kept the variables that have less or equal than 30% of missings and performed imputation of NA (which is, prediction of missing values) in order to gain information and better prediction. The imputation first was done for each country separatedly and if a country had all the observations missing for one variable we filled them with the mean of all the other countries.
- For the final step we used an algorithm called mutual information to select the variables that will be introduced in the model. The reason of this algorithm is because it's flexible (all variables have a value of importance, usually different from 0) and models non-linearities and complex high order interactions, as gradient boosting (the prediction model) does.
- We only keep the 50 variables with higher weight of the mutual information output. Among them, 3 are variables created for us: Time, Country, and lag1 (previous year of NY.GDP.MKTP.KD.ZG).

#### Extract and reshape data

The row data extracted from the database contains a set of columns with the structure of the database tables. This is reshaped into a Multi-index Dataframe where columns are Indicators and rows are Countries and Years. Rows with more NaN values than a specified threshold are eliminated. Then three more columns are added to this Dataframe: Country, Year and Predicted Indicator indicator Lag(1). The reason for this is that we believe those 3 indicators are key to perform a good fit of the model: 
  - Country is a kind of weight for the data. It is not the same having an indicator from Zimbabwe than from Switzerland, since the economy of the second one has a much stronger impact on the world's GDP growth. Countries are converted to numeric values to be used by the model as any other Indicator.
  
  - Year is also a kind of weight for time. As the economy grows exponentially, the year is an important Indicator because it weighs the response of the model depending on the year.  
  
  - Predicted Indicator Lag. We have seen that the set of possible Predicted Indicators, i.e. those that measure the growth in form percentage or difference, have a behavior similar to a memoryless process. Their autocorrelation function shows dependence only for lag(1), but a strong one. Using this information to train and predict is thus a must. 

#### Linear Model and Residuals

If we assume a linear fit of the data to predict the Predicted Indicator, we will incur a big mistake, since we know beforehand that its behavior is nonlinear. The reason to apply this step is to calculate the residuals from this linear fit, which are those non-linear counterparts that must be explained somehow.

#### Selection of Indicators

With the residuals from the linear model, we now use the mutual information regression to get a measure of each Indicator dependence. Then is just about selecting the top Indicators to use then in the final model training. 

#### TreeBoosting modelling

We use a  Gradient Tree Boosting model, combined with random effects (gpboost). For further details s. https://github.com/fabsig/GPBoost.
Our model can predict the GPD growth for any country and any year between 1961 and an arbitrary large year. Although, predictions for years that are too far in the future are not recommended.

Note:
Right now, the gpboost package cannot save properly the random effects part of the model. Also, most of the conventional analysis tools for ML algorithms, as e.g. SHAP, do not analyze the model correctly. Nonetheless, gpboost achieves higher accuracy than classic GBM models for the given panel data.
#### 

### HOW TO EXECUTE IT

The software is executed through the terminal or command prompt, but first, you need to install all required libraries.

1) Copy the project folder somewhere in your local machine.
2) CD into the folder GDP-Growth using the terminal or command prompt. 
3) Create a virtual enviroment and activate it: 
    virtualenv .venv
    source .venv/bin/activate
4) Install all libraries into that virtual environment:
    pip install -r requirements.txt
5) Now your local machine is ready to run the software. Type into the terminal the interpreter, followed by the software main file, and the year to be predicted:
    python cli.py predict --year 2011
6) The software will now create a model with the available data, and then predict the GDP-Growth for the year selected.
