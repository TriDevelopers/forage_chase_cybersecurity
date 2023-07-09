import pandas as pd
import matplotlib.pyplot as plt

def exercise_0(file):
    return pd.read_csv(file)

def exercise_1(df):
    return df.columns

def exercise_2(df, k):
    return df.head(k)

def exercise_3(df, k):
    return df.sample(k)

def exercise_4(df):
    unique = df['type'].unique()
    return unique

def exercise_5(df):
    top_des = df['nameDest'].value_counts()
    return top_des.head(10)

def exercise_6(df):
    flagged = df['isFlaggedFraud'] == 1
    return df[flagged]

def exercise_7(df):
    distinct_destinations=  df.groupby('nameOrig')['nameDest'].nunique().reset_index(name='num_destinations')
    sorted_distinct_destinations = distinct_destinations.sort_values(by='num_destinations', ascending=False)
    return sorted_distinct_destinations
    
def visual_1(df):
    def transaction_counts(df):
        counts = df['type'].value_counts()
        return counts

    def transaction_counts_split_by_fraud(df):
        counts = df.groupby(['type', 'isFraud']).size().unstack(fill_value=0)
        return counts

    fig, axs = plt.subplots(2, figsize=(6,10))
    transaction_counts(df).plot(ax=axs[0], kind='bar')
    axs[0].set_title('Transaction Counts by Type')
    axs[0].set_xlabel('Type')
    axs[0].set_ylabel('Count')
    transaction_counts_split_by_fraud(df).plot(ax=axs[1], kind='bar')
    axs[1].set_title('Transaction Counts Split by Fraud')
    axs[1].set_xlabel('Type')
    axs[1].set_ylabel('Count')
    fig.suptitle('Visualizations')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    for ax in axs:
      for p in ax.patches:
          ax.annotate(p.get_height(), (p.get_x(), p.get_height()))
    return 'The chart shows the counts of each type and which type has higher chance to contain fraud.'

def visual_2(df):
    def query(df):
        cash_out = df.query('type == "CASH_OUT"').copy()
        cash_out['origin_account_balance_delta'] = cash_out['newbalanceOrig'] - cash_out['oldbalanceOrg']
        cash_out['destination_account_balance_delta'] = cash_out['newbalanceDest'] - cash_out['oldbalanceDest']
        return cash_out
    plot = query(df).plot.scatter(x='origin_account_balance_delta', y='destination_account_balance_delta')
    plot.set_title('Origin Account Balance Delta v. Destination Account Balance Delta for CashOut type')
    plot.set_xlim(left=-1e3, right=1e3)
    plot.set_ylim(bottom=-1e3, top=1e3)
    return 'Origin Account Balance Delta v. Destination Account Balance Delta for CashOut type'

def exercise_custom(df):
    counts = df.groupby(['isFraud', 'isFlaggedFraud']).size().unstack(fill_value=0)
    return counts
    
def visual_custom(df):
    counts = exercise_custom(df)
    counts.plot.bar()
    plt.title('Detection Accuracy of Flagged Fraud Transactions')
    plt.xlabel('Flagged')
    plt.ylabel('Real Fraud')
    plt.show()