#!/bin/bash
set -e
Xvfb :1 -screen 0 1024x768x24 +extension GLX &
sleep 1
mkdir -p /var/run/sshd
exec /usr/sbin/sshd -D
