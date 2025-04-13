# Run this on your training data
import pandas as pd

df_train = pd.read_csv("./Dataset/train_fwYjLYX.csv")
# Group by date and segment
df_train = df_train.sort_values('application_date').groupby(
    ['application_date', 'segment'], as_index=False
).agg({'case_count': ['sum']})
df_train.columns = ['application_date', 'segment', 'case_count']

# Statistics by segment
for segment in [1, 2]:
  segment_data = df_train[df_train['segment'] == segment]
  print(f"\nSegment {segment} training data statistics:")
  print(segment_data['case_count'].describe())
