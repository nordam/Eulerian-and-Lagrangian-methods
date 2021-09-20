#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import datetime

def lagrangian_logger(message, args, error = False):
    log_entry = f'[{datetime.datetime.now()}] dt = {args.dt}, Np = {args.Np}, profile = {args.profile}, run = {args.run_id}   --   {message}'
    if args.verbose or error:
        if args.statusfilename is not None:
            args.statusfile.write(log_entry + '\n')
        else:
            print(log_entry)
            sys.stdout.flush()

def eulerian_logger(message, args, error = False):
    log_entry = f'[{datetime.datetime.now()}] dt = {args.dt}, NJ = {args.NJ}, NK = {args.NK}, profile = {args.profile}   --   {message}'
    if args.verbose or error:
        if args.statusfilename is not None:
            args.statusfile.write(log_entry + '\n')
        else:
            print(log_entry)
            sys.stdout.flush()

