from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_community.llms.fake import FakeListLLM
from langchain.prompts import PromptTemplate
from typing import Dict
import re

# Step 1 : Fraud Rules Engine
def fraud_score(transaction : Dict) -> Dict :
    score = 0
    reasons = []

    amount = transaction.get('amount', 0)
    location = transaction.get('location')
    home_location = transaction.get('home_location')
    device = transaction.get('device', "").lower()
    time = transaction.get('time', '')

    if amount > 1000 :
        score += 30 
        reasons.append('High transaction amount')
    
    if location and location.lower() and home_location != home_location.lower():
        score += 25
        reasons.append('Location mismatch')

    if 'new' in device:
        score += 20
        reasons.append('Unrecognized device')

    if time in ['1am', '2am', '3am', '4am'] :
        score += 15
        reasons.append('Suspicious time of transaction')

    risk = 'Low'
    if score >= 70:
        risk = 'High'
    elif score >= 50:
        risk = 'Medium'

    action = {
        'Low' : 'Approved',
        'Medium' : 'Require OTP',
        'High' : 'Decline and flag for review'
    }[risk]

    return {
        'score' : score,
        'risk_level' : risk,
        'recommended_action' : action,
        'reasons' : reasons
    }

# Step 2 : Fraud Scoring Tool Wrapper
def fraud_tool_fn(description : str) -> str :
    # naive parser
    amount = int(re.search(r"\$(\d+)", description).group(1))
    location = re.search(r"in (\w+)", description).group(1)
    home_location = re.search(r"user.*?in (\w+)", description).group(1)
    device = "new" if "new" in description else "known"
    time = re.search(r"at (\d+am|\d+pm)", description).group(1)

    transaction = {
        'amount' : amount,
        'location' : location,
        'home_location' : home_location,
        'device' : device,
        'time' : time
    }

    result = fraud_score(transaction)
    return f'Score : {result['score']}/100 \nRisk Level : {result['risk_level']} \nAction : {result['recommended_action']}\nReasons : {result['reasons']}'

fraud_tool = Tool(
    name="FraudScoringTool",
    func=fraud_tool_fn,
    description="Evaluates transaction descriptions and assigns fraud scores."
)

# Step 3 : Simulated LLM Responses
def simulate_llm(description) :
    return [
        "I should use the FraudScoringTool to evaluate this transaction.",
        fraud_tool_fn(description),
        "Based on the score and risk level, I have provided the best action."
    ]

# Step 4 : Main Agent Runner
def run_agent():
    print("\n🔒 Fraud Risk Agent (LangChain-based)\nType 'exit' to quit.\n")
    while True:
        user_input = input('Describe the trasnaction:\n> ')
        if user_input.lower() == 'exit':
            break

        llm = FakeListLLM(responses = simulate_llm(user_input))
        agent = initialize_agent(
            tools = [fraud_tool],
            llm = llm,
            agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose = True
        )

        try :
            output = agent.run(user_input)
            print(f'\n✅ Agent Decision:\n{output}\n'),
        except Exception as e :
            print(f"❌ Error: {e}\n")

if __name__ == '__main__':
    run_agent()

