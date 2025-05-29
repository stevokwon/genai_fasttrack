import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import csv
from fraud_risk_agent import run_fraud_risk_agent, fraud_tool_fn

# Initialize an empty list to score transaction results
if "transaction_history" not in st.session_state:
    st.session_state.transaction_history = []

# Feature 1 : Visualisation of Fraud Score over Time
def plot_transaction_history():
    if st.session_state.transaction_history:
        df = pd.DataFrame(st.session_state.transaction_history)
        df["Transaction"] = df["Transaction"].apply(lambda x : x[:25] + '...' if len(x) > 25 else x)
        fig, ax = plt.subplots()
        ax.plot(df["Transaction"], df["Fraud Score"], marker='o')
        ax.set_xlabel("Transaction Summary")
        ax.set_ylabel("Fraud Score")
        ax.set_title("Fraud Score Over Time")
        st.pyplot(fig)
    else:
        st.info("No transactions to plot yet.")

# Feature 2 : Transaction History Table
def update_transaction_history(transaction, score, risk, action):
    st.session_state.transaction_history.append({
        "Transaction": transaction,
        "Fraud Score": score,
        "Risk Level": risk,
        "Action": action
    })

# Feature 3 : Show Deatiled Breakdown of Reasons
def display_detailed_breakdown(description: str):
    result = fraud_tool_fn(description)  # Run the agent on the transaction description
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
            writer.writerow([
                result.get("Transaction", ""), 
                result.get("Fraud Score", ""), 
                result.get("Risk Level", ""), 
                result.get("Action", "")
            ])

# Step 1 : Set up the Streamlit UI
def run_streamlit_ui():
    st.title('🔒 Fraud Risk Agent')
    st.write("""
        Welcome to the **Fraud Risk Agent** powered by LangChain.

        ### Steps:
        1. **Describe the transaction** in the input box.
        2. **Click Evaluate** to get the fraud score, action, and reasons.
        3. **View fraud score history** over time.
        4. **Download results** as a CSV file for further analysis.
    """)

    # Input box to describe the transaction
    user_input = st.text_area("Describe the transaction", "")

    if st.button("Evaluate"):
        if user_input:
            output = fraud_tool_fn(user_input)  # Run the agent on user input
            st.markdown("### ✅ Agent Decision")
            st.write(f"**Fraud Score:** {output['score']}/100")
            st.write(f"**Risk Level:** {output['risk_level']}")
            st.write(f"**Recommended Action:** {output['recommended_action']}")

            if output.get("reasons"):
                st.write("**Reasons:**")
                for reason in output["reasons"]:
                    st.markdown(f"- {reason}")

            # Update transaction history
            update_transaction_history(user_input, output['score'], output['risk_level'], output['recommended_action'])

            # ✅ Success message
            st.success("Transaction evaluated and added to history")

            # ✅ Clear input
            st.session_state['user_input'] = ""

        else:
            st.warning("Please enter a transaction description.")

    # Feature 1: Show fraud score history
    if st.button("Show Fraud Score History"):
        plot_transaction_history()

    # Feature 2: Show transaction history
    if st.button("Show Transaction History"):
        if st.session_state.transaction_history:
            st.dataframe(pd.DataFrame(st.session_state.transaction_history))
        else:
            st.info("No transaction history yet.")

    # Feature 3: Show detailed breakdown
    if st.button("Show Detailed Breakdown"):
        if user_input:
            breakdown = display_detailed_breakdown(user_input)
            st.write(breakdown)
        else:
            st.warning("Please enter a transaction description.")

    # Feature 4: Download results as CSV
    if st.button("Download Results"):
        save_results_to_csv(st.session_state.transaction_history)
        st.success("Results saved successfully!")

if __name__ == '__main__':
    run_streamlit_ui()