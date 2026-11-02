#!/bin/sh
# Revert llms.txt reading view centering + Last-updated bump (Jacob pt1063u). Restores website/llms-page.html and llms.txt as of 20261102T151952Z.
set -e
cp -p /home/const/subnet120-rewrite/state/bench-results/site-backup/20261102T151952Z/llms-page.html /home/const/subnet120-rewrite/state/bench-results/site-backup/20261102T151952Z/llms.txt /home/const/subnet120/affine/website/
echo reverted llms page to 20261102T151952Z
