# tau2-gen telecom - fidelity report

## Task level

| metric | generated | reference |
|---|---|---|
| n | 200 | - |
| intents | {'mms_issue': '86 (43%)', 'mobile_data_issue': '63 (32%)', 'service_issue': '51 (26%)'} | - |
| personas | {'Easy': '39 (20%)', 'Hard': '51 (26%)', 'Impatient': '17 (8%)', 'NonNative': '8 (4%)', 'None': '44 (22%)', 'SideRequest': '5 (2%)', 'TechSavvy': '6 (3%)', 'Terse': '12 (6%)', 'Verbose': '10 (5%)', 'WrongNumberOnce': '8 (4%)'} | - |
| n_faults | {1: '21 (10%)', 2: '41 (20%)', 3: '36 (18%)', 4: '24 (12%)', 5: '30 (15%)', 6: '20 (10%)', 7: '11 (6%)', 8: '12 (6%)', 9: '5 (2%)'} | - |
| expected_actions_min_med_max | (1, 4, 11) | - |
| write_actions_min_med_max | (0, 4, 11) | - |
| no_write_tasks | 35 (18%) | - |
| unfixable | 35 (18%) | - |
| reward_basis | {'ENV_ASSERTION|ACTION': 35, 'ENV_ASSERTION': 165} | - |
| assertion_funcs | {'assert_can_send_mms': 86, 'assert_mobile_data_status': 63, 'assert_internet_speed': 63, 'assert_data_refueling_amount': 58, 'assert_service_status': 51, 'assert_no_overdue_bill': 45} | - |
| init_funcs | {'set_user_info': 200, 'turn_airplane_mode_on': 95, 'set_user_location': 93, 'turn_data_off': 90, 'set_data_usage': 87, 'simulate_network_search': 77, 'set_network_mode_preference': 77, 'unseat_sim_card': 75, 'remove_app_permission': 71, 'disable_roaming': 58, 'turn_roaming_off': 57, 'set_wifi_calling': 49, 'break_apn_mms_setting': 48, 'turn_data_saver_mode_on': 39, 'turn_roaming_on': 36, 'enable_roaming': 35, 'break_vpn': 33, 'break_apn_settings': 32, 'refuel_data': 29, 'suspend_line_for_overdue_bill': 19, 'lock_sim_card': 4} | - |
| with_initialization_data | 200 | - |

## Leakage

- Result: **PASS**. Identifier overlap {'phones': 0, 'ids': 0, 'emails': 0, 'imeis': 0, 'names': 0}. Held-out task-id overlap 0 (out of 114 held-out tasks).
- Informational: 30 tasks share a (composition, persona) with tau2's full enumeration of 2285. Those are not held out, so they are allowed.

## Known false-fail rates

_No must-mention strings in this set._

## Rollout level

_No --gen-traces given: run a reference model over the set first (see README, "Reference rollouts"). The table below only shows the reference side._

| metric | generated | reference | threshold | verdict |
|---|---|---|---|---|
| reference pass rate | - | - | [0.6, 0.85] | n.a. |
| tasks with >=2 usable rollouts | - | - | >= 0.9 | n.a. |
| tasks with non-identical rollouts | - | - | >= 0.8 | n.a. |
