"""Telecom curated actions derived only from native visible goals/instructions.

No database or evaluator imports. User actions depend on actual visible tool
responses and the agent's public instruction. Candidate-computation policy is
explicit; this is not an unbiased historical Engy user simulator.
"""
import json

VERSION='native-telecom-public-instruction-candidates-v1'
POSITIVE_GUIDANCE='Please check your network status. If roaming is disabled, enable roaming on your phone. Then run a speed test and tell me whether the speed is excellent.'
NEGATIVE_GUIDANCE='Please run a speed test and tell me whether the speed is excellent.'

def tool(name):return json.dumps({'tool_call':{'name':name,'arguments':{}}},sort_keys=True)

def visible_tool_results(request):
    return [m.get('content','') for m in request['messages'] if m.get('role')=='tool']

def public_action(request,agent_guidance=POSITIVE_GUIDANCE):
    role={'native-model-agent':'agent','native-model-user':'user'}.get(request.get('model'))
    if role is None:raise ValueError('public policy role')
    names={t['function']['name'] for t in request.get('tools',[])}
    if role=='agent':return agent_guidance,'agent-public-diagnostic-guidance'
    tools=[m for m in request['messages'] if m.get('role')=='tool']
    agent_messages=[m.get('content','') for m in request['messages'] if m.get('role')=='user']
    # Tau2 flips the user-role conversation: the support agent's words are user.
    instructions='\n'.join(agent_messages).lower()
    if not tools and len(agent_messages)<=1:
        return 'My mobile data is not working well while I am abroad. I want excellent internet speed. Please help.','user-public-goal'
    if not tools:
        name='check_network_status' if 'check your network status' in instructions else 'run_speed_test'
        if name not in names:raise ValueError('missing native public tool')
        return tool(name),'agent-requested-public-diagnostic'
    last=tools[-1];last_name=last.get('name','')
    if not last_name:
        for message in request['messages']:
            for call in message.get('tool_calls') or []:
                if call.get('id')==last.get('tool_call_id'):last_name=call.get('function',{}).get('name',call.get('name',''))
    content=str(last.get('content',''));lower=content.lower()
    if last_name=='check_network_status':
        # All decisions use actual tool output, never an expected answer/state.
        disabled=('data roaming enabled: no' in lower or 'roaming_enabled: false' in lower or 'roaming_enabled": false' in lower or 'roaming is disabled' in lower or 'roaming is off' in lower or 'roaming: disabled' in lower or 'roaming status: disabled' in lower)
        if disabled and 'enable roaming' in instructions:
            if 'toggle_roaming' not in names:raise ValueError('missing native roaming tool')
            return tool('toggle_roaming'),'visible-disabled-roaming-plus-agent-instruction'
        return tool('run_speed_test'),'requested-speed-test-after-visible-network-state'
    if last_name=='toggle_roaming':return tool('run_speed_test'),'requested-speed-test-after-actual-roaming-action'
    if last_name=='run_speed_test':
        if 'excellent' in lower:return 'The actual speed test reports excellent internet speed. My issue is resolved. ###STOP###','observed-goal-completion'
        return 'The actual speed test is not excellent. My issue is not resolved; please help.','observed-unsolved-goal'
    return 'Please clarify what I should do next.','public-clarification'

def format_candidates(action):
    # Two equivalent serializations expose actual model preference without
    # forcing an unproved output; all selected strings are explicitly curated.
    try:
        obj=json.loads(action)
    except ValueError:return [action,action+' ']
    return [json.dumps(obj,sort_keys=True,separators=(',',':')),json.dumps(obj,sort_keys=True)]
