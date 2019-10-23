import pandas as pd

data = pd.read_csv("data/emp_attrition.csv")

# replace binary labels with 1's and 0's
binary_data = {'Gender': ['Male', 'Female'],
               'Over18': ['N', 'Y'],
               'OverTime': ['No', 'Yes']
               }

for k, v in binary_data.items():
    print('{}, {}', k, v)
    data[k].replace(v, [0, 1], inplace=True)

data.rename(columns={'Gender': 'IsFemale'}, inplace=True)

one_hot_cols = ['BusinessTravel',
                'Department',
                'EducationField',
                'JobRole',
                'MaritalStatus'
                ]

for col_name in one_hot_cols:
    # one-hot encode certain column
    data = pd.concat([data, pd.get_dummies(data[col_name], drop_first=True)],
                     axis=1
                     )

    # drop original column
    data.drop([col_name], axis=1, inplace=True)
print(data)
