from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.llms.fake import FakeListLLM
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