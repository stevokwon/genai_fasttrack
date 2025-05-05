from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_openai import ChatOpenAI
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
    amount_match = re.search(r"\$(\d+)", description)
    location_match = re.search(r"in (\w+)", description)
    home_location_match = re.search(r"user.*?in (\w+)", description)
    time_match = re.search(r"at (\d+am|\d+pm)", description)

    amount = int(amount_match.group(1)) if amount_match else 0
    location = location_match.group(1) if location_match else "Unknown"
    home_location = home_location_match.group(1) if home_location_match else "Unknown"
    device = "new" if "new" in description.lower() else "known"
    time = time_match.group(1) if time_match else "Unknown"

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

# --- Step 3: Prompt Template ---
fraud_prompt = PromptTemplate.from_template("""
You are a fraud detection assistant working for a financial institution.

Your task is to:
1. Understand a natural language description of a transaction.
2. Decide whether to invoke a fraud scoring tool.
3. If you invoke the tool, always provide the Action Input with the following information:
   - amount (e.g. $1200)
   - transaction location (e.g. in Paris)
   - time (e.g. at 2am)
   - device used (e.g. new iPhone)
   - user's home location (e.g. user in Singapore)

Use this format:
Thought: [Your reasoning]
Action: FraudScoringTool
Action Input: "A $<amount> transaction in <location> at <time> using <device> from a user in <home_location>"

Begin!

{input}
""")

fraud_tool = Tool(
    name="FraudScoringTool",
    func=fraud_tool_fn,
    description="Evaluates transaction descriptions and assigns fraud scores."
)

# Step 3 : Main Agent Runner
def run_agent():
    print("\n🔒 Fraud Risk Agent (LangChain-based)\nType 'exit' to quit.\n")
    while True:
        user_input = input('Describe the trasnaction:\n> ')
        if user_input.lower() == 'exit':
            break

        llm = ChatOpenAI(temperature = 0, model = 'gpt-3.5-turbo')
        agent = initialize_agent(
            tools = [fraud_tool],
            llm = llm,
            agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            agent_kwargs = {'prompt' : fraud_prompt},
            verbose = True
        )

        try :
            output = agent.run(user_input)
            print(f'\n✅ Agent Decision:\n{output}\n'),
        except Exception as e :
            print(f"❌ Error: {e}\n")

if __name__ == '__main__':
    run_agent()

