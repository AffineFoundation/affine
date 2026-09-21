# tau2-gen telecom - fidelity report

## Task level

| metric | generated | reference |
|---|---|---|
| n | 200 | - |
| intents | {'mms_issue': '86 (43%)', 'mobile_data_issue': '63 (32%)', 'service_issue': '51 (26%)'} | - |
| personas | {'Easy': '40 (20%)', 'Hard': '39 (20%)', 'Impatient': '12 (6%)', 'NonNative': '13 (6%)', 'None': '41 (20%)', 'SideRequest': '6 (3%)', 'TechSavvy': '10 (5%)', 'Terse': '14 (7%)', 'Verbose': '13 (6%)', 'WrongNumberOnce': '12 (6%)'} | - |
| n_faults | {1: '4 (2%)', 2: '44 (22%)', 3: '46 (23%)', 4: '38 (19%)', 5: '28 (14%)', 6: '13 (6%)', 7: '10 (5%)', 8: '8 (4%)', 9: '9 (4%)'} | - |
| expected_actions_min_med_max | (1, 4, 12) | - |
| write_actions_min_med_max | (0, 4, 12) | - |
| no_write_tasks | 35 (18%) | - |
| unfixable | 35 (18%) | - |
| reward_basis | {'ENV_ASSERTION|ACTION': 35, 'ENV_ASSERTION': 165} | - |
| assertion_funcs | {'assert_can_send_mms': 86, 'assert_data_refueling_amount': 66, 'assert_mobile_data_status': 63, 'assert_internet_speed': 63, 'assert_service_status': 51, 'assert_no_overdue_bill': 34} | - |
| init_funcs | {'set_user_info': 200, 'turn_airplane_mode_on': 106, 'turn_data_off': 87, 'set_data_usage': 84, 'set_network_mode_preference': 82, 'simulate_network_search': 80, 'unseat_sim_card': 78, 'set_user_location': 75, 'remove_app_permission': 70, 'break_apn_mms_setting': 52, 'disable_roaming': 48, 'set_wifi_calling': 48, 'turn_roaming_off': 42, 'turn_data_saver_mode_on': 40, 'turn_roaming_on': 33, 'suspend_line_for_overdue_bill': 32, 'break_vpn': 32, 'break_apn_settings': 32, 'enable_roaming': 27, 'refuel_data': 18, 'lock_sim_card': 15} | - |
| with_initialization_data | 200 | - |

## Leakage

- Result: **PASS**. Identifier overlap {'phones': 0, 'ids': 0, 'emails': 0, 'imeis': 0, 'names': 0}. Held-out task-id overlap 0 (out of 114 held-out tasks).
- Informational: 22 tasks share a (composition, persona) with tau2's full enumeration of 2285. Those are not held out, so they are allowed.

## Known false-fail rates

_No must-mention strings in this set._

## Rollout level

_No --gen-traces given: run a reference model over the set first (see README, "Reference rollouts"). The table below only shows the reference side._

| metric | generated | reference | threshold | verdict |
|---|---|---|---|---|
| reference pass rate | - | - | [0.6, 0.85] | n.a. |
| tasks with >=2 usable rollouts | - | - | >= 0.9 | n.a. |
| tasks with non-identical rollouts | - | - | >= 0.8 | n.a. |
