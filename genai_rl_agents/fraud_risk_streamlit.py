import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import csv
from fraud_risk_agent import run_fraud_risk_agent

# Initialize an empty list to score transaction results
transaction_history = []

# 1. Visualisation of Fraud Score over Time
def plot_transaction_history():
    if transaction_history:
        df = pd.DataFrame(transaction_history, columns = ['Transaction', 'Fraud Score', 'Risk Level'])
        fig, ax = plt.subplots()
        ax.plot(df['Transaction'], df['Fraud Score'], marker = 'o')
        ax.set_xlabel('Transaction')
        ax.set_ylabel('Fraud Score')
        ax.set_title('Fraud Score Over Time')
        st.pyplot(fig)

# 2. Transaction History Table
def update_transaction_history(transaction, score, risk, action):
    global transaction_history
    transaction_history = transaction_history.append({
        'Transaction' : transaction,
        'Fraud Score' : score,
        'Risk Level' : risk,
        'Action' : action
    }, ignore_index = True)

# Step 1 : Set up the Streamlit UI
def run_streamlit_ui():
    st.title('🔒 Fraud Risk Agent')
    st.write('Type a transaction description to get the fraud risk assessment.')

    # Text input for transaction description
    user_input = st.text_area('Describe the transaction', '')

    if st.button('Evaluate'):
        if user_input:
            output = run_fraud_risk_agent(user_input)
            st.success(f'✅ Agent Decision: \n{output}')
        else:
            st.warning('Please enter a transaction description')
    
if __name__ == '__main__':
    run_streamlit_ui()