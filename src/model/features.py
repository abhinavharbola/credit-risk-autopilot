"""Single source of truth for column names. Everything else imports from here
instead of restating strings, so a typo in a column name breaks at one place.
"""

TARGET = "SeriousDlqin2yrs"

FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

# columns with missing values, median-imputed using training-pool-only medians
IMPUTE_COLUMNS = ["MonthlyIncome", "NumberOfDependents"]

ALL_COLUMNS = [TARGET] + FEATURES
