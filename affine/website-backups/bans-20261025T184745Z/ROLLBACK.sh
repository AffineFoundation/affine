#!/bin/sh
# Roll back the Banned miners section (restores v7 index.html + grid.css, removes bans.js/bans.json and the refresh timer)
set -e
W=/home/const/subnet120/affine/website; B=/home/const/subnet120/affine/website-backups/bans-20261025T184745Z
cp $B/index.html $W/.index.html.new && mv $W/.index.html.new $W/index.html
cp $B/grid.css $W/.grid.css.new && mv $W/.grid.css.new $W/grid.css
rm -f $W/bans.js $W/bans.json
systemctl --user disable --now affine-site-bans.timer 2>/dev/null || true
rm -f ~/.config/systemd/user/affine-site-bans.service ~/.config/systemd/user/affine-site-bans.timer
systemctl --user daemon-reload
echo rolled back
