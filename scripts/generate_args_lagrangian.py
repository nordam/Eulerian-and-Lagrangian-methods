#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Install the default signal handler.
# The purpose is to not get an error when piping the output to, e.g., head
# See https://stackoverflow.com/questions/14207708/ioerror-errno-32-broken-pipe-when-piping-prog-py-othercmd
from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE, SIG_DFL)

Nps = [3000000, 1000000, 300000, 100000, 30000, 10000, 3000, 1000, 300, 100]
dts = [2, 10, 20, 60, 100, 150, 300, 600, 1800, 3600]
profiles = ['B',]
Nruns = 100


dt = 10
for Np in Nps:
    for profile in profiles:
        for n in range(Nruns):
            print(f'--Np {Np} --dt {dt} --profile {profile} --run_id {n}')

Np = 3000000
for dt in dts:
    for profile in profiles:
        for n in range(Nruns):
            print(f'--Np {Np} --dt {dt} --profile {profile} --run_id {n}')

