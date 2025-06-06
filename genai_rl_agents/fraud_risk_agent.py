from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from typing import Dict
import re
import json
import os

# Step 1 : Load External Scoring Rules
RULES_PATH = os.path.join(os.path.dirname(__file__), 'fraud_rules.json')
with open(RULES_PATH, 'r') as f:
    FRAUD_RULES = json.load(f)['rules']

# Step 2 : Fraud Rules Engine
def fraud_score(transaction : Dict) -> Dict :
    score = 0
    reasons = []

    for rule in FRAUD_RULES:
        try:
            condition = rule['condition']
            if eval(condition, {}, transaction):
                score += rule['score']
                reasons.append(rule['reason'])
        except Exception as e:
            reasons.append(f"Error in the rule '{condition}' : {e}")

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

# Step 3 : Fraud Scoring Tool Wrapper
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
    return {
        "score": result['score'],
        "risk_level": result['risk_level'],
        "recommended_action": result['recommended_action'],
        "reasons": result['reasons'],
        "raw_output": f"Score: {result['score']}/100\nRisk Level: {result['risk_level']}\nAction: {result['recommended_action']}\nReasons: {', '.join(result['reasons'])}"
    }

fraud_tool = Tool(
    name="FraudScoringTool",
    func=fraud_tool_fn,
    description="Evaluates transaction descriptions and assigns fraud scores."
)

# --- Step 4: Prompt Template ---
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

# Step 5: Run Fraud Risk Agent
def run_fraud_risk_agent(description: str) -> Dict:
    raw_result = fraud_tool_fn(description)

    try:
        score_part = raw_result.split("Score: ")[1].split("/")[0]
        score = int(score_part)
        risk = raw_result.split("Risk Level: ")[1].split("\n")[0]
        action = raw_result.split("Action: ")[1].split("\n")[0]
        reasons_section = raw_result.split("Reasons: ")[1].strip()
        reasons = [r.strip() for r in reasons_section.split(",") if r.strip()]
    except (IndexError, ValueError) as e:
        # In case the string parsing fails
        return {
            "score": 0,
            "risk_level": "Unknown",
            "recommended_action": "Unknown",
            "reasons": ["Failed to extract result details."],
            "raw_output": raw_result  # Optional: useful for debugging
        }

    return {
        "score": score,
        "risk_level": risk,
        "recommended_action": action,
        "reasons": reasons,
        "raw_output": raw_result
    }

# Step 6: Main Agent Runner
def run_agent():
    print("\n🔒 Fraud Risk Agent (LangChain-based)\nType 'exit' to quit.\n")
    while True:
        user_input = input("Describe the transaction:\n> ")
        if user_input.lower() == 'exit':
            break

        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
        agent = initialize_agent(
            tools=[fraud_tool],
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            agent_kwargs={"prompt": fraud_prompt},
            verbose=True
        )
        try:
            output = agent.invoke({"input": user_input})
            print(f"\n✅ Agent Decision:\n{output}\n")
        except Exception as e:
            print(f"❌ Error: {e}\n")

if __name__ == "__main__":
    run_agent()