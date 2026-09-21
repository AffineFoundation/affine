# tau2-gen retail - fidelity report

## Task level

| metric | generated | reference |
|---|---|---|
| n | 200 | 114 |
| intents | {'address_then_payment': '4 (2%)', 'cancel_delivered_denied': '4 (2%)', 'cancel_one_return_other': '11 (6%)', 'cancel_pending': '7 (4%)', 'cancel_two_orders': '9 (4%)', 'cancel_unknown_order': '6 (3%)', 'composed:cancel_denied+exchange_ok': '2 (1%)', 'composed:cancel_denied+exchange_ok+return_ok': '1 (0%)', 'composed:cancel_denied+modify_address_ok': '3 (2%)', 'composed:cancel_denied+modify_items_ok': '2 (1%)', 'composed:cancel_denied+modify_payment_denied_balance+profile_address_ok': '1 (0%)', 'composed:cancel_denied+modify_payment_ok': '3 (2%)', 'composed:cancel_denied+modify_payment_ok+profile_address_ok': '1 (0%)', 'composed:cancel_denied+profile_address_ok': '4 (2%)', 'composed:cancel_denied+return_ok': '1 (0%)', 'composed:cancel_ok+exchange_ok': '1 (0%)', 'composed:cancel_ok+exchange_ok+modify_items_cross_denied': '2 (1%)', 'composed:cancel_ok+exchange_ok+modify_items_ok': '1 (0%)', 'composed:cancel_ok+exchange_ok+modify_payment_denied_balance': '1 (0%)', 'composed:cancel_ok+exchange_ok+modify_payment_ok': '2 (1%)', 'composed:cancel_ok+exchange_ok+profile_address_ok': '1 (0%)', 'composed:cancel_ok+modify_address_ok': '2 (1%)', 'composed:cancel_ok+modify_address_ok+return_pending_denied': '1 (0%)', 'composed:cancel_ok+modify_items_cross_denied': '2 (1%)', 'composed:cancel_ok+modify_items_cross_denied+return_pending_denied': '3 (2%)', 'composed:cancel_ok+modify_items_ok': '1 (0%)', 'composed:cancel_ok+modify_items_ok+modify_payment_ok': '1 (0%)', 'composed:cancel_ok+modify_items_ok+return_ok': '1 (0%)', 'composed:cancel_ok+modify_items_ok+return_pending_denied': '1 (0%)', 'composed:cancel_ok+modify_payment_ok': '1 (0%)', 'composed:cancel_ok+modify_payment_ok+profile_address_ok': '1 (0%)', 'composed:cancel_ok+profile_address_ok': '3 (2%)', 'composed:cancel_ok+profile_address_ok+return_ok': '1 (0%)', 'composed:exchange_ok+modify_address_ok': '1 (0%)', 'composed:exchange_ok+modify_address_ok+modify_payment_ok': '1 (0%)', 'composed:exchange_ok+modify_address_ok+return_pending_denied': '1 (0%)', 'composed:exchange_ok+modify_items_cross_denied': '2 (1%)', 'composed:exchange_ok+modify_items_ok': '2 (1%)', 'composed:exchange_ok+modify_items_ok+modify_payment_denied_balance': '1 (0%)', 'composed:exchange_ok+modify_payment_denied_balance+profile_address_ok': '1 (0%)', 'composed:exchange_ok+modify_payment_ok+profile_address_ok': '1 (0%)', 'composed:exchange_ok+profile_address_ok': '1 (0%)', 'composed:exchange_ok+profile_address_ok+return_ok': '1 (0%)', 'composed:exchange_ok+return_ok': '3 (2%)', 'composed:exchange_ok+return_pending_denied': '4 (2%)', 'composed:modify_address_ok+modify_items_cross_denied+return_pending_denied': '1 (0%)', 'composed:modify_address_ok+modify_items_ok': '2 (1%)', 'composed:modify_address_ok+modify_items_ok+return_ok': '1 (0%)', 'composed:modify_address_ok+modify_payment_ok': '3 (2%)', 'composed:modify_address_ok+modify_payment_ok+return_pending_denied': '2 (1%)', 'composed:modify_address_ok+return_ok': '1 (0%)', 'composed:modify_address_ok+return_pending_denied': '1 (0%)', 'composed:modify_items_cross_denied+modify_payment_ok': '1 (0%)', 'composed:modify_items_cross_denied+modify_payment_ok+return_ok': '1 (0%)', 'composed:modify_items_ok+modify_payment_denied_balance': '2 (1%)', 'composed:modify_items_ok+profile_address_ok': '1 (0%)', 'composed:modify_items_ok+return_ok': '2 (1%)', 'composed:modify_items_ok+return_pending_denied': '2 (1%)', 'composed:modify_payment_denied_balance+profile_address_ok': '1 (0%)', 'composed:modify_payment_denied_balance+profile_address_ok+return_pending_denied': '1 (0%)', 'composed:modify_payment_ok+profile_address_ok+return_pending_denied': '1 (0%)', 'composed:modify_payment_ok+return_ok': '3 (2%)', 'composed:profile_address_ok+return_ok': '6 (3%)', 'composed:profile_address_ok+return_pending_denied': '1 (0%)', 'exchange_delivered': '7 (4%)', 'exchange_then_profile_address': '4 (2%)', 'exchange_two_items': '4 (2%)', 'exchange_unavailable_denied': '3 (2%)', 'modify_address': '5 (2%)', 'modify_delivered_denied': '3 (2%)', 'modify_items': '7 (4%)', 'modify_items_cross_product_denied': '3 (2%)', 'modify_payment': '4 (2%)', 'modify_user_address': '4 (2%)', 'other_user_denied': '2 (1%)', 'return_delivered': '7 (4%)', 'return_pending_denied': '3 (2%)', 'return_unknown_order': '7 (4%)'} | {'cancel_pending_order': '6 (5%)', 'exchange_delivered_order_items': '12 (11%)', 'find_user_id_by_email': '11 (10%)', 'find_user_id_by_name_zip': '55 (48%)', 'modify_pending_order_address': '10 (9%)', 'modify_pending_order_items': '7 (6%)', 'no_action': '2 (2%)', 'return_delivered_order_items': '10 (9%)', 'transfer_to_human_agents': '1 (1%)'} |
| personas | {'Easy': '16 (8%)', 'Hard': '9 (4%)', 'Impatient': '16 (8%)', 'NonNative': '8 (4%)', 'None': '98 (49%)', 'SideRequest': '10 (5%)', 'TechSavvy': '7 (4%)', 'Terse': '11 (6%)', 'Verbose': '16 (8%)', 'WrongNumberOnce': '9 (4%)'} | {'None': '114 (100%)'} |
| n_faults | {1: '200 (100%)'} | {None: '114 (100%)'} |
| expected_actions_min_med_max | (3, 6, 10) | (0, 5, 13) |
| write_actions_min_med_max | (0, 1, 3) | (0, 1, 5) |
| no_write_tasks | 16 (8%) | 10 (9%) |
| unfixable | 0 (0%) | 4 (4%) |
| reward_basis | {'DB|NL_ASSERTION': 200} | {'DB|NL_ASSERTION': 112, 'DB': 2} |
| assertion_funcs | {} | {} |
| init_funcs | {} | {} |
| with_initialization_data | 200 | 0 |

## Leakage

- Result: **PASS**. Identifier overlap {'phones': 0, 'ids': 0, 'emails': 0, 'imeis': 0, 'names': 0}. Held-out task-id overlap 0 (out of 114 held-out tasks).
- Informational: 0 tasks share a (composition, persona) with tau2's full enumeration of 114. Those are not held out, so they are allowed.

## Known false-fail rates

_Must-mention strings are recorded but do not gate: COMMUNICATE is not in the reward basis. Strings in use: {'cancel': 81, 'exchange': 48, 'address': 63, 'payment': 34, 'return': 58, 'item': 42, 'order': 2}._

## Rollout level

_No --gen-traces given: run a reference model over the set first (see README, "Reference rollouts"). The table below only shows the reference side._

| metric | generated | reference | threshold | verdict |
|---|---|---|---|---|
| reference pass rate | - | - | [0.6, 0.85] | n.a. |
| tasks with >=2 usable rollouts | - | - | >= 0.9 | n.a. |
| tasks with non-identical rollouts | - | - | >= 0.8 | n.a. |
