#!/bin/bash
# usage: pod.sh probe|trainer|eval "cmd"   (read-only helper; BatchMode, strict host keys)
case "$1" in
 probe) H=38.88.142.89; P=40198; K=b4f3da9d-df66-436a-af38-829d1606966d;;
 trainer) H=8.28.35.244; P=40022; K=ef2b6085-adae-43e2-8a13-39182a3b3f7a;;
 eval) H=38.255.28.20; P=20092; K=6e469697-f3a7-4a6a-a1ae-40dde7120581;;
esac
shift
exec ssh -T -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=15 -o UserKnownHostsFile=/home/const/.lium/known_hosts/$K -p $P root@$H "$@"
