#!/usr/bin/env python
"""
DMLC submission script by ssh

One need to make sure all slaves machines are ssh-able.
"""
from __future__ import absolute_import

import os, subprocess, logging
from threading import Thread
from . import tracker

def sync_dir(local_dir, slave_node, slave_dir):
    """
    sync the working directory from root node into slave node
    """
    remote = slave_node[0] + ':' + slave_dir
    logging.info('rsync %s -> %s', local_dir, remote)
    prog = 'rsync -az --rsh="ssh -o StrictHostKeyChecking=no -p %s" %s %s' % (
        slave_node[1], local_dir, remote)
    subprocess.check_call([prog], shell = True)

def get_env(pass_envs):
    envs = []
    # get system envs
    keys = ['OMP_NUM_THREADS', 'KMP_AFFINITY', 'LD_LIBRARY_PATH', 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY']
    for k in keys:
        v = os.getenv(k)
        if v is not None:
            envs.append('export ' + k + '=' + v + ';')
    # get ass_envs
    for k, v in pass_envs.items():
        envs.append('export ' + str(k) + '=' + str(v) + ';')
    return (' '.join(envs))

def submit(args):
    print(args)
    assert args.host_file is not None
    with open(args.host_file) as f:
        tmp = f.readlines()
    assert len(tmp) > 0
    hosts=[]
    deviceIDs=[]
    for h in tmp:
        if len(h.strip()) > 0:
            # parse addresses of the form ip:port
            h = h.strip()
            items = h.split( )
            h = items[0]
            d = items[1]
            i = h.find(":")
            p = "22"
            if i != -1:
                p = h[i+1:]
                h = h[:i]
            # hosts now contain the pair ip, port
            hosts.append((h, p))
            deviceIDs.append(d)

    def ssh_submit(nworker, nserver, pass_envs):
        """
        customized submit script
        """
        # thread func to run the job
        def run(prog):
            subprocess.check_call(prog, shell = True)

        # sync programs if necessary
        local_dir = os.getcwd()+'/'
        working_dir = local_dir
        if args.sync_dst_dir is not None and args.sync_dst_dir != 'None':
            working_dir = args.sync_dst_dir
            for h in hosts:
                sync_dir(local_dir, h, working_dir)

        # launch jobs
        for i in range(nworker + nserver):
            pass_envs['DMLC_ROLE'] = 'server' if i < nserver else 'worker'
            (node, port) = hosts[i % len(hosts)]
            deviceID = deviceIDs[i % len(hosts)]
            if pass_envs['DMLC_ROLE'] == 'server':
                working_dir = args.s_dir
                deviceID = 0
                prog = 'source ~/anaconda2/bin/activate;' + get_env(pass_envs) + ' cd ' + working_dir + '; ' + (' '.join(args.command)) +  ' --gid ' + (' '.join(str(deviceID)))
                prog = 'ssh -o StrictHostKeyChecking=no ' + node + ' -p ' + port + ' \'' + prog + '\''
            else:
                working_dir = args.w_dir
                prog = 'source ~/anaconda2/bin/activate; ' + get_env(pass_envs) + ' cd ' + working_dir + '; ' + (' '.join(args.command)) + ' --gid ' + (' '.join(str(deviceID))) \
                        + ' --wid ' + str(i-nserver);
                prog = 'ssh -o StrictHostKeyChecking=no ' + node + ' -p ' + port + ' \'' + prog + '\''

            print prog
            thread = Thread(target = run, args=(prog,))
            thread.setDaemon(True)
            thread.start()

        return ssh_submit

    tracker.submit(args.num_workers, args.num_servers,
                   fun_submit=ssh_submit,
                   pscmd=(' '.join(args.command)))
