import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import csv
from fraud_risk_agent import run_fraud_risk_agent

# Initialize an empty list to score transaction results
transaction_history = []

# Feature 1 : Visualisation of Fraud Score over Time
def plot_transaction_history():
    if transaction_history:
        df = pd.DataFrame(transaction_history, columns = ['Transaction', 'Fraud Score', 'Risk Level'])
        fig, ax = plt.subplots()
        ax.plot(df['Transaction'], df['Fraud Score'], marker = 'o')
        ax.set_xlabel('Transaction')
        ax.set_ylabel('Fraud Score')
        ax.set_title('Fraud Score Over Time')
        st.pyplot(fig)

# Feature 2 : Transaction History Table
def update_transaction_history(transaction, score, risk, action):
    global transaction_history
    transaction_history = transaction_history.append({
        'Transaction' : transaction,
        'Fraud Score' : score,
        'Risk Level' : risk,
        'Action' : action
    }, ignore_index = True)

# Feature 3 : Show Deatiled Breakdown of Reasons
def fraud_tool_fn(description : str) -> str:
    result = run_fraud_risk_agent(description)
    breakdown = f"Transaction Details:\n{description}\n\n"
    breakdown += f"Score: {result['score']}/100\nRisk Level: {result['risk_level']}\n"
    breakdown += f"Action: {result['recommended_action']}\n"
    breakdown += "Reasons:\n"
    for reason in result["reasons"]:
        breakdown += f"- {reason}\n"
    return breakdown

# Feature 4 : Save and Download Results as CSV
def save_results_to_csv(results):
    with open('fraud_detection_results.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['Transaction', 'Fraud Score', 'Risk Level', 'Action'])
        for result in results:
            writer.writerow([result['transaction'], result['score'], result['risk'], result['action']])

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
    
    # Feature 1 : Show fraud score history
    if st.button('Show Fraud Score History'):
        plot_transaction_history()
    
    # Feature 2 : Show transaction history
    if st.button('Show Transaction History'):
        update_transaction_history()
    
    # Feature 3 : Show detailed breakdown
    if st.button('Show Detailed Breakdown'):
        fraud_tool_fn()
    
    # Feature 4 : Download results to CSV
    if st.button('Download Results'):
        save_results_to_csv()

if __name__ == '__main__':
    run_streamlit_ui()