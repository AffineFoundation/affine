#!/bin/bash
# Install the ArbosLife box hardening (pm2 under systemd + needrestart guard +
# dead-man timer). Idempotent. Run as const with passwordless sudo:
#
#   ~/subnet120/ops/box/install.sh
#
# It NEVER restarts pm2-const or the validator. If the pm2 daemon is running
# outside the unit (e.g. after a fresh box), start the unit by hand at a duel
# boundary: `sudo systemctl start pm2-const` adopts the live daemon without
# restarting the apps (see pm2-const-start).
#
# Files (source -> destination):
#   pm2-const.service.d/10-affine-adopt.conf -> /etc/systemd/system/pm2-const.service.d/
#   pm2-const-start                          -> /usr/local/sbin/pm2-const-start
#   needrestart-pm2-const.conf               -> /etc/needrestart/conf.d/50-pm2-const.conf
#   affine-deadman                           -> /usr/local/sbin/affine-deadman
#   affine-deadman.service, .timer           -> /etc/systemd/system/
#
# Pre-conditions this checks: /etc/systemd/system/pm2-const.service exists
# (create it with `pm2 startup systemd -u const --hp /home/const` if not) and
# `loginctl` reports Linger=yes for const.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_NAME=const

if [ ! -f /etc/systemd/system/pm2-const.service ]; then
  echo "install.sh: /etc/systemd/system/pm2-const.service is missing." >&2
  echo "  run: sudo env PATH=\$PATH pm2 startup systemd -u $USER_NAME --hp /home/$USER_NAME" >&2
  exit 1
fi

sudo install -d -m 0755 /etc/systemd/system/pm2-const.service.d /etc/needrestart/conf.d
sudo install -m 0644 -o root -g root "$HERE/pm2-const.service.d/10-affine-adopt.conf" /etc/systemd/system/pm2-const.service.d/10-affine-adopt.conf
sudo install -m 0755 -o root -g root "$HERE/pm2-const-start" /usr/local/sbin/pm2-const-start
sudo install -m 0644 -o root -g root "$HERE/needrestart-pm2-const.conf" /etc/needrestart/conf.d/50-pm2-const.conf
sudo install -m 0755 -o root -g root "$HERE/affine-deadman" /usr/local/sbin/affine-deadman
sudo install -m 0644 -o root -g root "$HERE/affine-deadman.service" /etc/systemd/system/affine-deadman.service
sudo install -m 0644 -o root -g root "$HERE/affine-deadman.timer" /etc/systemd/system/affine-deadman.timer
sudo install -d -m 0755 -o "$USER_NAME" -g "$USER_NAME" "/home/$USER_NAME/.affine"

sudo systemd-analyze verify pm2-const.service affine-deadman.service affine-deadman.timer
sudo systemctl daemon-reload
sudo systemctl enable pm2-const.service >/dev/null
sudo systemctl enable --now affine-deadman.timer >/dev/null
sudo loginctl enable-linger "$USER_NAME"

echo "pm2-const:      enabled=$(systemctl is-enabled pm2-const) active=$(systemctl is-active pm2-const)"
echo "deadman timer:  enabled=$(systemctl is-enabled affine-deadman.timer) active=$(systemctl is-active affine-deadman.timer)"
echo "linger:         $(loginctl show-user "$USER_NAME" -p Linger --value)"
echo "needrestart:    $(sudo needrestart -b -r l 2>/dev/null | grep -c 'pm2-const' || true) pm2-const line(s) pending (0 is expected)"
echo "deadman status: $(sudo /usr/local/sbin/affine-deadman --status)"
