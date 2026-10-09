"""Acquire fresh training selection only after successor calibration completes.

CPU orchestration only: the original calibration, assessment acquisition,
ROOT authentication, and freshness validators remain unchanged.
"""
VERSION = 'assessment-after-successor-calibration-v1'


def install(selection, calibration, *, earliest_round):
    if type(earliest_round) is not int or earliest_round < 1:
        raise ValueError('explicit future opening assessment boundary')
    original_prepare = selection.prepare_opening
    original_calibrate = calibration.before_open

    def prospective(status):
        value = status.get('round')
        if type(value) is not int or value < 0:
            raise ValueError('exact opening round')
        return value >= earliest_round

    def fresh_only(contract):
        if 'start' in contract or 'deadline' in contract:
            raise ValueError('issued manifest assessment must remain immutable')

    def prepare_opening(controller, config, status, contract):
        if not prospective(status):
            return original_prepare(controller, config, status, contract)
        fresh_only(contract)
        # No cached snapshot or unsigned placeholder: original acquisition runs
        # below only after the exact calibration/ACK chain returns successfully.
        return contract

    def before_open(controller, config, status, contract):
        if not prospective(status):
            return original_calibrate(controller, config, status, contract)
        fresh_only(contract)
        calibrated = original_calibrate(controller, config, status, contract)
        fresh_only(calibrated)
        return original_prepare(controller, config, status, calibrated)

    selection.prepare_opening = prepare_opening
    calibration.before_open = before_open
    return dict(prepare_opening=original_prepare, before_open=original_calibrate)
