# tau2-gen telecom - fidelity report

## Task level

| metric | generated | reference |
|---|---|---|
| n | 200 | - |
| intents | {'mms_issue': '86 (43%)', 'mobile_data_issue': '63 (32%)', 'service_issue': '51 (26%)'} | - |
| personas | {'Easy': '33 (16%)', 'Hard': '42 (21%)', 'Impatient': '11 (6%)', 'NonNative': '17 (8%)', 'None': '41 (20%)', 'SideRequest': '15 (8%)', 'TechSavvy': '6 (3%)', 'Terse': '8 (4%)', 'Verbose': '15 (8%)', 'WrongNumberOnce': '12 (6%)'} | - |
| n_faults | {1: '23 (12%)', 2: '50 (25%)', 3: '35 (18%)', 4: '27 (14%)', 5: '27 (14%)', 6: '14 (7%)', 7: '13 (6%)', 8: '6 (3%)', 9: '5 (2%)'} | - |
| expected_actions_min_med_max | (1, 3, 11) | - |
| write_actions_min_med_max | (0, 3, 11) | - |
| no_write_tasks | 35 (18%) | - |
| unfixable | 35 (18%) | - |
| reward_basis | {'ENV_ASSERTION': 165, 'ENV_ASSERTION|ACTION': 35} | - |
| assertion_funcs | {'assert_can_send_mms': 86, 'assert_mobile_data_status': 63, 'assert_internet_speed': 63, 'assert_data_refueling_amount': 55, 'assert_service_status': 51, 'assert_no_overdue_bill': 47} | - |
| init_funcs | {'set_user_info': 200, 'turn_airplane_mode_on': 96, 'set_data_usage': 86, 'set_user_location': 86, 'set_network_mode_preference': 82, 'turn_data_off': 76, 'simulate_network_search': 71, 'remove_app_permission': 61, 'unseat_sim_card': 61, 'disable_roaming': 57, 'turn_roaming_off': 57, 'set_wifi_calling': 49, 'break_apn_mms_setting': 41, 'turn_data_saver_mode_on': 36, 'break_vpn': 35, 'refuel_data': 31, 'break_apn_settings': 30, 'turn_roaming_on': 29, 'enable_roaming': 29, 'suspend_line_for_overdue_bill': 14, 'lock_sim_card': 3} | - |
| with_initialization_data | 200 | - |

## Leakage

- Result: **PASS**. Identifier overlap {'phones': 0, 'ids': 0, 'emails': 0, 'imeis': 0, 'names': 0}. Held-out task-id overlap 0 (out of 114 held-out tasks).
- Informational: 34 tasks share a (composition, persona) with tau2's full enumeration of 2285. Those are not held out, so they are allowed.

## Known false-fail rates

_No must-mention strings in this set._

## Rollout level

_No --gen-traces given: run a reference model over the set first (see README, "Reference rollouts"). The table below only shows the reference side._

| metric | generated | reference | threshold | verdict |
|---|---|---|---|---|
| reference pass rate | - | - | [0.6, 0.85] | n.a. |
| tasks with >=2 usable rollouts | - | - | >= 0.9 | n.a. |
| tasks with non-identical rollouts | - | - | >= 0.8 | n.a. |
