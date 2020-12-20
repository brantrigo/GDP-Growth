import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATABASE_PATH = os.path.join(BASE_DIR, "db.sqlite3")

MODELS_PATH = os.path.join(BASE_DIR, "models")

LOGS_PATH = os.path.join(BASE_DIR, "logs")

PREDICTED_INDICATOR = 'NY.GDP.MKTP.KD.ZG'

exclude_list = ['Arab World', 'Caribbean small states', 'Central Europe and the Baltics',
 'East Asia & Pacific \(all income levels',
 'East Asia & Pacific \(developing only', 'Euro area',
 'Europe & Central Asia \(all income levels',
 'Europe & Central Asia \(developing only', 'European Union',
 'Fragile and conflict affected situations',
 'Heavily indebted poor countries \(HIPC', 'High income',
 'High income: nonOECD', 'High income: OECD',
 'Latin America & Caribbean \(all income levels',
 'Latin America & Caribbean \(developing only',
 'Least developed countries: UN classification', 'Low & middle income',
 'Low income', 'Lower middle income',
 'Middle East & North Africa \(all income levels',
 'Middle East & North Africa \(developing only', 'Middle income',
 'North America' 'OECD members' ,'Other small states',
 'Pacific island small states', 'Small states', 'South Asia',
 'Sub-Saharan Africa \(all income levels',
 'Sub-Saharan Africa \(developing only' ,'Upper middle income' ,'World', 'North America', 'OECD members']
 
DB_YEAR_MIN = 1960

DB_YEAR_MAX = 2010
