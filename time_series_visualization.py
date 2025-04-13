# Plot predictions over time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

submission = pd.read_csv("submission.csv")
submission['application_date'] = pd.to_datetime(submission['application_date'])

# Plot by segment
plt.figure(figsize=(14, 8))
for segment in [1, 2]:
    segment_data = submission[submission['segment'] == segment]
    plt.subplot(2, 1, segment)
    plt.plot(segment_data['application_date'], segment_data['case_count'])
    plt.title(f'Predicted Case Count for Segment {segment}')
    plt.ylabel('Case Count')
    plt.grid(True)

plt.tight_layout()
plt.show()
