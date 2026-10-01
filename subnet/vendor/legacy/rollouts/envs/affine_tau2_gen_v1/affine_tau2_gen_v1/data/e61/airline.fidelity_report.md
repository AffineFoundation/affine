# tau2-gen airline - fidelity report

## Task level

| metric | generated | reference |
|---|---|---|
| n | 400 | 50 |
| intents | {'baggage_add': '6 (2%)', 'baggage_remove_denied': '20 (5%)', 'book': '7 (2%)', 'cancel_airline_cancelled': '5 (1%)', 'cancel_business': '5 (1%)', 'cancel_denied': '41 (10%)', 'cancel_denied_flown': '25 (6%)', 'cancel_insurance_health': '5 (1%)', 'cancel_then_compensation': '5 (1%)', 'cancel_two_reservations': '12 (3%)', 'cancel_unknown_reservation': '9 (2%)', 'cancel_within_24h': '5 (1%)', 'change_cabin': '7 (2%)', 'change_flights': '9 (2%)', 'change_flights_denied_basic_economy': '41 (10%)', 'change_flights_then_baggage': '5 (1%)', 'compensation_cancelled_flight': '4 (1%)', 'compensation_denied': '25 (6%)', 'composed:baggage_add+cancel_denied': '1 (0%)', 'composed:baggage_add+cancel_denied+cancel_denied_flown': '1 (0%)', 'composed:baggage_add+cancel_denied+change_cabin_ok': '1 (0%)', 'composed:baggage_add+cancel_denied+change_flights_denied': '1 (0%)', 'composed:baggage_add+cancel_denied+compensation_ok': '1 (0%)', 'composed:baggage_add+cancel_denied_flown+insurance_denied': '1 (0%)', 'composed:baggage_add+cancel_ok': '1 (0%)', 'composed:baggage_add+cancel_ok+compensation_denied': '2 (0%)', 'composed:baggage_add+change_cabin_ok': '1 (0%)', 'composed:baggage_add+change_cabin_ok+change_flights_ok': '1 (0%)', 'composed:baggage_add+change_cabin_ok+compensation_ok': '1 (0%)', 'composed:baggage_add+change_flights_denied': '1 (0%)', 'composed:baggage_add+change_flights_denied+insurance_denied': '1 (0%)', 'composed:baggage_add+change_flights_ok': '2 (0%)', 'composed:baggage_add+compensation_denied': '1 (0%)', 'composed:baggage_add+compensation_denied+insurance_denied': '1 (0%)', 'composed:baggage_add+compensation_ok': '3 (1%)', 'composed:baggage_add+insurance_denied': '2 (0%)', 'composed:baggage_add+passenger_change_ok': '1 (0%)', 'composed:baggage_add+passenger_count_denied': '2 (0%)', 'composed:baggage_remove_denied+cancel_denied+change_flights_ok': '2 (0%)', 'composed:baggage_remove_denied+cancel_denied+passenger_change_ok': '1 (0%)', 'composed:baggage_remove_denied+cancel_denied_flown+change_flights_ok': '1 (0%)', 'composed:baggage_remove_denied+cancel_ok': '2 (0%)', 'composed:baggage_remove_denied+cancel_ok+passenger_change_ok': '1 (0%)', 'composed:baggage_remove_denied+change_cabin_ok': '3 (1%)', 'composed:baggage_remove_denied+change_flights_ok': '3 (1%)', 'composed:baggage_remove_denied+change_flights_ok+passenger_change_ok': '1 (0%)', 'composed:baggage_remove_denied+change_flights_ok+passenger_count_denied': '1 (0%)', 'composed:baggage_remove_denied+compensation_ok+passenger_change_ok': '2 (0%)', 'composed:baggage_remove_denied+insurance_denied+passenger_change_ok': '1 (0%)', 'composed:baggage_remove_denied+passenger_change_ok': '2 (0%)', 'composed:cancel_denied+cancel_denied_flown+change_flights_ok': '1 (0%)', 'composed:cancel_denied+cancel_denied_flown+passenger_change_ok': '1 (0%)', 'composed:cancel_denied+change_cabin_ok': '2 (0%)', 'composed:cancel_denied+change_cabin_ok+passenger_count_denied': '2 (0%)', 'composed:cancel_denied+change_flights_ok+compensation_ok': '2 (0%)', 'composed:cancel_denied+compensation_ok': '4 (1%)', 'composed:cancel_denied+compensation_ok+passenger_count_denied': '1 (0%)', 'composed:cancel_denied+passenger_change_ok': '3 (1%)', 'composed:cancel_denied_flown+cancel_ok': '1 (0%)', 'composed:cancel_denied_flown+cancel_ok+change_cabin_ok': '1 (0%)', 'composed:cancel_denied_flown+cancel_ok+change_flights_denied': '1 (0%)', 'composed:cancel_denied_flown+cancel_ok+passenger_change_ok': '1 (0%)', 'composed:cancel_denied_flown+change_cabin_ok': '1 (0%)', 'composed:cancel_denied_flown+change_cabin_ok+change_flights_ok': '1 (0%)', 'composed:cancel_denied_flown+change_flights_ok': '2 (0%)', 'composed:cancel_denied_flown+compensation_ok': '1 (0%)', 'composed:cancel_denied_flown+passenger_change_ok': '2 (0%)', 'composed:cancel_ok+change_cabin_ok': '2 (0%)', 'composed:cancel_ok+change_cabin_ok+change_flights_ok': '1 (0%)', 'composed:cancel_ok+change_flights_denied': '2 (0%)', 'composed:cancel_ok+insurance_denied': '2 (0%)', 'composed:cancel_ok+passenger_change_ok': '1 (0%)', 'composed:cancel_ok+passenger_count_denied': '1 (0%)', 'composed:change_cabin_ok+change_flights_ok': '3 (1%)', 'composed:change_cabin_ok+change_flights_ok+compensation_ok': '1 (0%)', 'composed:change_cabin_ok+compensation_denied': '1 (0%)', 'composed:change_cabin_ok+compensation_denied+insurance_denied': '1 (0%)', 'composed:change_cabin_ok+compensation_ok': '2 (0%)', 'composed:change_cabin_ok+compensation_ok+insurance_denied': '1 (0%)', 'composed:change_cabin_ok+insurance_denied': '2 (0%)', 'composed:change_cabin_ok+passenger_count_denied': '3 (1%)', 'composed:change_flights_denied+compensation_ok': '3 (1%)', 'composed:change_flights_denied+compensation_ok+insurance_denied': '1 (0%)', 'composed:change_flights_denied+passenger_change_ok': '1 (0%)', 'composed:change_flights_ok+compensation_denied': '2 (0%)', 'composed:change_flights_ok+compensation_denied+passenger_change_ok': '1 (0%)', 'composed:change_flights_ok+compensation_ok+passenger_count_denied': '1 (0%)', 'composed:change_flights_ok+passenger_change_ok': '1 (0%)', 'composed:change_flights_ok+passenger_count_denied': '1 (0%)', 'composed:compensation_ok+insurance_denied': '1 (0%)', 'composed:compensation_ok+insurance_denied+passenger_change_ok': '1 (0%)', 'composed:compensation_ok+insurance_denied+passenger_count_denied': '1 (0%)', 'composed:compensation_ok+passenger_count_denied': '2 (0%)', 'composed:insurance_denied+passenger_change_ok': '2 (0%)', 'insurance_add_denied': '20 (5%)', 'passenger_change': '4 (1%)', 'passenger_count_denied': '20 (5%)', 'upgrade_then_baggage': '6 (2%)'} | {'book_reservation': '4 (8%)', 'cancel_reservation': '3 (6%)', 'get_reservation_details': '14 (28%)', 'get_user_details': '13 (26%)', 'no_action': '7 (14%)', 'search_direct_flight': '1 (2%)', 'transfer_to_human_agents': '1 (2%)', 'update_reservation_flights': '7 (14%)'} |
| personas | {'Easy': '37 (9%)', 'Hard': '24 (6%)', 'Impatient': '18 (4%)', 'NonNative': '20 (5%)', 'None': '210 (52%)', 'SideRequest': '11 (3%)', 'TechSavvy': '11 (3%)', 'Terse': '23 (6%)', 'Verbose': '30 (8%)', 'WrongNumberOnce': '16 (4%)'} | {'None': '50 (100%)'} |
| n_faults | {1: '400 (100%)'} | {None: '50 (100%)'} |
| expected_actions_min_med_max | (2, 4, 8) | (0, 2, 19) |
| write_actions_min_med_max | (0, 1, 3) | (0, 1, 5) |
| no_write_tasks | 192 (48%) | 24 (48%) |
| unfixable | 41 (10%) | 1 (2%) |
| reward_basis | {'DB|COMMUNICATE': 400} | {'DB|COMMUNICATE': 50} |
| assertion_funcs | {} | {} |
| init_funcs | {} | {} |
| with_initialization_data | 400 | 0 |

## Leakage

- Result: **PASS**. Identifier overlap {'phones': 0, 'ids': 0, 'emails': 0, 'imeis': 0, 'names': 0}. Held-out task-id overlap 0 (out of 50 held-out tasks).
- Informational: 0 tasks share a (composition, persona) with tau2's full enumeration of 50. Those are not held out, so they are allowed.

## Known false-fail rates

| must-mention string | tasks | measured recall | known false-fail |
|---|---|---|---|
| `cancel` | 164 | anchored on the user's own words | not separately measured |
| `flight` | 94 | anchored on the user's own words | not separately measured |
| `bag` | 72 | anchored on the user's own words | not separately measured |
| `passenger` | 62 | anchored on the user's own words | not separately measured |
| `basic economy` | 52 | 11/12 (92 %) | 8 % of the 52 tasks carrying it |
| `cabin` | 44 | anchored on the user's own words | not separately measured |
| `insurance` | 38 | anchored on the user's own words | not separately measured |
| `certificate` | 34 | anchored on the user's own words | not separately measured |
| `compensation` | 33 | anchored on the user's own words | not separately measured |
| `book` | 7 | anchored on the user's own words | not separately measured |

A correct agent that phrases its answer without the string fails the task. That is a property of the check, not of the agent, so it is reported here and not folded into the pass rate above.

## Rollout level

_No --gen-traces given: run a reference model over the set first (see README, "Reference rollouts"). The table below only shows the reference side._

| metric | generated | reference | threshold | verdict |
|---|---|---|---|---|
| reference pass rate | - | - | [0.6, 0.85] | n.a. |
| tasks with >=2 usable rollouts | - | - | >= 0.9 | n.a. |
| tasks with non-identical rollouts | - | - | >= 0.8 | n.a. |
