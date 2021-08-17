#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import datetime

def logger(message, args, error = False):
    log_entry = f'[{datetime.datetime.now()}] dt = {args.dt}, Np = {args.Np}, run = {args.run_id:04}, profile = {args.profile}   --   {message}'
    if args.verbose or error:
        if args.statusfilename is not None:
            args.statusfile.write(log_entry + '\n')
        else:
            print(log_entry)
            sys.stdout.flush()

