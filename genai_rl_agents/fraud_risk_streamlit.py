import streamlit as st
from fraud_risk_agent import run_fraud_risk_agent

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