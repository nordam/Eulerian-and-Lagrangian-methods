#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Install the default signal handler.
# The purpose is to not get an error when piping the output to, e.g., head
# See https://stackoverflow.com/questions/14207708/ioerror-errno-32-broken-pipe-when-piping-prog-py-othercmd
from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE, SIG_DFL)

NKs = [256, 128, 96, 72, 64, 48, 36, 32, 24, 16, 12, 8, 4]
NJs = [16000, 12000, 8000, 6000, 4000, 3000, 2000, 1500, 1000, 750, 500]
dts = [5, 10, 20, 60, 100, 150, 300, 600, 1800]
profiles = ['B',]


dt = 10
NJ = 8000
for NK in NKs:
    for profile in profiles:
        print(f'--NK {NK} --dt {dt} --NJ {NJ} --profile {profile}')

dt = 10
NK = 128
for NJ in NJs:
    for profile in profiles:
        print(f'--NK {NK} --dt {dt} --NJ {NJ} --profile {profile}')

NK = 128
NJ = 8000
for dt in dts:
    for profile in profiles:
        print(f'--NK {NK} --dt {dt} --NJ {NJ} --profile {profile}')
