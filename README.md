## GDP Growth Forecast

Bran Alfeiran Trigo 1588728
Ferran Paüls Vergés 1393184
Kevin Yanes García 1590310

### INTRODUCTION

Gross domestic product (GDP) is a monetary measure of the market value
of all the final goods and services produced in a specific time period. GDP is
often used as a metric for international comparisons as well as a broad measure
of economic progress. It is often considered to be the "world’s most powerful
statistical indicator of national development and progress".
You are provided with a dataset containing over a thousand annual indicators of economic development (including GDP and GDP growth) in hundreds of
countries around the world from 1960 to 2010, and you are asked to develop a
model that can predict GDP growth using data from the past. You can use all the
variables that you want from previous years, and you can use the model that you
feel it is going to work better. The aim is to have a model that can predict next
year GDP growth for each country in the world using the available data.

### FUNCTIONAL AND TECHNICAL DESCRIPTION OF THE SOFTWARE

The software predicts the GDP-Growth of all available countries into a dataset, for the year specified by the user. 

#### Extract and reshape data

The row data extracted from database contains a set of columns with the structure of the database tables. This is reshaped into a Multi-index Dataframe where columns are Indicators and rows are Contries and Years. Rows with more NaN values than a specified threshold are eliminated. Then three more columns are added to this Dataframe: Country, Year and Predicted Indicator indicator Lag(1). The reason for this is that we believe those 3 indicators are key to perform a good fit of the model: 
  - Country is kind of a weigh for the data. It is not the same having an indicator from Zimbabwe than from Switzerland, since the economy of the second one has a much stronger impact in the worlds GDP Growth. Countries are converted to numeric values to be used by the model as any other Indicator.
  
  - Year is also a kind of weigh for time. As the economy grows explonentially, the year is an important Indicator because it weighs the response of the model depending on the year.  
  
  - Predicted Indicator Lag. We have seen that the set of possible Predicted Indicators, i.e. those that measure the growth in form percentage or difference, have a behaviour similar to a memoryless process. Their autocorrelation function shows dependence only for lag(1), but a strong one. Using this information to train and predict is thus a must. 

#### Linear Model and Residuals

If we assume a linear fit of the data to predict the Predicted Indicator, we will incur into a big mistake, since we know beforhand that its behaviour is non.linear. The reason to apply this step is to calculate the residuals from this linear fit, which are thos non-linear counterparts that must be explained somehow.

#### Selection of Indicators

With the residuals from the linear model, we now use the mutual information regression to get a measure of each Indicator dependence. Then is just about selecting the top Indicators to use then in the final model trainning. 

#### TreeBoosting modelling

#### 

### HOW TO EXECUTE IT

The software is executed through the terminal or command prompt, but first you need to install all required libraries.

1) Copy the project folder somewehre in your local machine.
2) CD into the folder GDP-Growth using the terminal or command prompt. 
3) Create a virtual enviroment and activate it: 
    virtualenv .venv
    source .venv/bin/activate
4) Install all libraries into that virtual enviroment:
    pip install -r requirements.txt
5) Now you local machine is ready to run the software. Type into the terminal the interpreter, followed by the software main file, and the year to be predicted:
    python cli.py predict 2011
6) The software will now create a model with the available data, and then predict the GDP-Growth for the year selected.


