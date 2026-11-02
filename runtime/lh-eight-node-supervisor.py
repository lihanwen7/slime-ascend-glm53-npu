#!/usr/bin/env python3
"""Deadline-bound compute-window guard and launcher; standard library only."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    os.replace(temporary, path)


def identity(pid):
    try:
        root = Path('/proc') / str(pid)
        raw = (root / 'stat').read_text()
        rest = raw[raw.rfind(')') + 2:].split()
        return {'pid': pid, 'ticks': rest[19], 'ppid': int(rest[1]), 'state': rest[0],
                'cmd': (root / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')}
    except (OSError, ValueError, IndexError):
        return None


def alive(row):
    current = identity(row['pid'])
    return current is not None and current['ticks'] == row['ticks'] and current['state'] != 'Z'


def signal_identity(row, sig):
    if alive(row):
        try:
            os.kill(row['pid'], sig)
            return True
        except ProcessLookupError:
            pass
    return False


def processes():
    result = {}
    for entry in Path('/proc').iterdir():
        if entry.name.isdigit():
            row = identity(int(entry.name))
            if row is not None and row['state'] != 'Z':
                result[row['pid']] = row
    return result


def npu_snapshot():
    try:
        proc = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True, timeout=12)
        text = proc.stdout
        devices = []
        last = None
        pids = []
        for line in text.splitlines():
            header = re.match(r'\|\s*(\d+)\s*\|\s*(Ascend\S+)\s*\|\s*(\S+)\s*\|', line)
            if header:
                last = {'device': int(header[1]), 'name': header[2], 'health': header[3]}
            elif last is not None and 'NA' in line:
                pairs = re.findall(r'(\d+)\s*/\s*(\d+)', line)
                if pairs:
                    last.update(hbm_used_mib=int(pairs[-1][0]), hbm_total_mib=int(pairs[-1][1]))
                    devices.append(last)
                    last = None
            match = re.match(r'\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([^|]*)\|\s*(\d+)\s*\|', line)
            if match:
                process_identity = identity(int(match[2]))
                pids.append({'device': int(match[1]), 'pid': int(match[2]),
                             'ticks': process_identity['ticks'] if process_identity else None,
                             'name': match[3].strip(), 'memory_mib': int(match[4])})
        return {'returncode': proc.returncode, 'devices': devices, 'processes': pids,
                'stderr': proc.stderr[-1000:], 'raw': text}
    except Exception as exc:
        return {'error': str(exc), 'devices': [], 'processes': []}


def memory_snapshot(rows, owned):
    host = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        if key in ('MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree', 'Mlocked', 'Unevictable'):
            host[key + '_kib'] = int(value.split()[0])
    rss = 0
    locked = 0
    for pid, row in owned.items():
        if pid not in rows or rows[pid]['ticks'] != row['ticks']:
            continue
        try:
            for line in (Path('/proc') / str(pid) / 'status').read_text().splitlines():
                if line.startswith('VmRSS:'):
                    rss += int(line.split()[1])
                elif line.startswith('VmLck:'):
                    locked += int(line.split()[1])
        except OSError:
            pass
    host.update(task_rss_sum_kib=rss, task_locked_sum_kib=locked)
    return host


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', required=True, type=Path)
    parser.add_argument('--node', required=True, type=int)
    parser.add_argument('--remaining-seconds', required=True, type=float)
    args = parser.parse_args()
    cfg = json.loads(args.protocol.read_text())
    root = args.protocol.parent
    node = root / ('node' + str(args.node))
    node.mkdir(exist_ok=True)
    lock = open('/tmp/glm53-exclusive-window.lock', 'a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    owner = cfg['owner']
    os.environ['GLM53_EXCLUSIVE_OWNER'] = owner
    if args.node not in range(21, 29):
        raise ValueError('Only nodes 177.2.21.1 through 177.2.28.1 are authorized')
    addresses = subprocess.check_output(['ip', '-o', '-4', 'addr'], text=True)
    if f'177.2.{args.node}.1/' not in addresses:
        raise ValueError('Node address does not match the authorized 177.2 host')
    remaining = min(args.remaining_seconds, cfg['deadline_epoch'] - time.time())
    deadline = time.monotonic() + max(0, remaining)
    stop_at = deadline - cfg.get('cleanup_lead_seconds', 90)
    me = identity(os.getpid())
    write_json(node / 'guard-identity.json', dict(me, owner=owner, boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                                                deadline_local_epoch=cfg['deadline_epoch']))
    events = open(node / 'guard-events.jsonl', 'a', buffering=1)
    samples = open(node / 'memory.jsonl', 'a', buffering=1)
    own = {me['pid']: me}
    pending = {}
    launched = set()
    child = None
    child_row = None
    active_attempt = None
    last_npu = {}
    last_sample = 0
    stopping = False
    shutdown_reason = None
    first_stop = None
    next_foreign_scan = 0
    markers = tuple(cfg['compute_markers'])
    ownership_marker = ('GLM53_EXCLUSIVE_OWNER=' + owner).encode()
    terminate = False

    def requested_stop(signum, frame):
        nonlocal terminate
        terminate = True

    signal.signal(signal.SIGTERM, requested_stop)
    signal.signal(signal.SIGINT, requested_stop)

    def event(kind, **values):
        events.write(json.dumps(dict(epoch=time.time(), event=kind, **values), ensure_ascii=False) + '\n')

    def terminate_rows(selected, reason):
        for row in selected:
            key = (row['pid'], row['ticks'])
            if row['pid'] <= 1 or row['pid'] == me['pid'] or key in pending:
                continue
            if signal_identity(row, signal.SIGTERM):
                pending[key] = (time.monotonic() + 5, row)
                event('terminate', reason=reason, identity=row)

    while time.monotonic() < deadline:
        now = time.monotonic()
        rows = processes()
        for pid, row in rows.items():
            if pid in own and own[pid]['ticks'] == row['ticks']:
                own[pid] = row
                continue
            parent = own.get(row['ppid'])
            inherited = parent is not None and row['ppid'] in rows and rows[row['ppid']]['ticks'] == parent['ticks']
            if not inherited:
                try:
                    inherited = ownership_marker in (Path('/proc') / str(pid) / 'environ').read_bytes().split(b'\0')
                except OSError:
                    inherited = False
            if inherited:
                own[pid] = row
        current_own = {pid: row for pid, row in own.items() if pid in rows and rows[pid]['ticks'] == row['ticks']}
        for key, (kill_at, row) in list(pending.items()):
            if not alive(row):
                del pending[key]
            elif now >= kill_at:
                if signal_identity(row, signal.SIGKILL):
                    event('kill', identity=row)
                del pending[key]
        if now - last_sample >= cfg.get('sample_seconds', 15):
            last_sample = now
            last_npu = npu_snapshot()
            sample = {'epoch': time.time(), 'host_memory': memory_snapshot(rows, current_own),
                      'npu': {key: value for key, value in last_npu.items() if key != 'raw'},
                      'owned_processes': len(current_own)}
            available = sample['host_memory'].get('MemAvailable_kib', 0)
            sample['alerts'] = []
            if available < 64 * 1024 * 1024:
                sample['alerts'].append('host_available_below_64GiB')
            if any(d.get('hbm_used_mib', 0) > d.get('hbm_total_mib', 1) * 0.97 for d in last_npu.get('devices', [])):
                sample['alerts'].append('hbm_used_above_97percent')
            samples.write(json.dumps(sample) + '\n')
            write_json(node / 'memory-latest.json', sample)
            (node / 'npu-latest.txt').write_text(last_npu.get('raw', ''))
            write_json(node / 'owned-identities.json', list(current_own.values()))
            if available < 16 * 1024 * 1024:
                shutdown_reason = 'host_available_below_16GiB'
                write_json(root / ('emergency-node' + str(args.node) + '.json'), sample)
        if now >= next_foreign_scan and now < stop_at:
            next_foreign_scan = now + 2
            foreign = {pid: row for pid, row in rows.items() if pid not in current_own and any(m in row['cmd'] for m in markers)}
            for entry in last_npu.get('processes', []):
                pid = entry['pid']
                if pid in rows and pid not in current_own and rows[pid]['ticks'] == entry.get('ticks'):
                    foreign[pid] = rows[pid]
            # Stop descendants of identified compute controllers without climbing
            # into SSH sessions, container engines or system-service ancestors.
            for _ in range(16):
                added = {pid: row for pid, row in rows.items() if row['ppid'] in foreign and pid not in current_own}
                foreign.update(added)
            terminate_rows(foreign.values(), 'exclusive_compute_window')
        if not stopping and (now >= stop_at or terminate or (root / 'stop.json').exists() or shutdown_reason):
            stopping = True
            shutdown_reason = shutdown_reason or ('deadline' if now >= stop_at else 'requested_stop')
            first_stop = now
            event('stopping', reason=shutdown_reason)
        try:
            retire = json.loads((root / 'retire.json').read_text())
        except FileNotFoundError:
            retire = None
        retiring = retire is not None and retire.get('run_id') == active_attempt
        if stopping:
            terminate_rows([row for pid, row in current_own.items() if pid != me['pid']], shutdown_reason)
            if now - first_stop > 12 and len(current_own) == 1:
                break
        elif retiring:
            terminate_rows([row for pid, row in current_own.items() if pid != me['pid']], 'retire_attempt')
        elif (root / 'launch.json').exists():
            request = json.loads((root / 'launch.json').read_text())
            attempt = request['run_id']
            if attempt not in launched:
                if child is None or (child.poll() is not None and len(current_own) == 1):
                    launched.add(attempt)
                    active_attempt = attempt
                    launch_env = dict(os.environ, **request.get('env', {}))
                    output = open(node / (attempt + '-launcher.log'), 'ab', buffering=0)
                    child = subprocess.Popen(['bash', request['script'], '--run-id', attempt] + request.get('args', []),
                                             env=launch_env, stdout=output, stderr=subprocess.STDOUT,
                                             start_new_session=True)
                    child_row = identity(child.pid)
                    own[child.pid] = child_row
                    write_json(node / 'launch-identity.json', dict(child_row, attempt=attempt))
                    event('launch', attempt=attempt, identity=child_row)
        child_code = child.poll() if child is not None else None
        write_json(node / 'state.json', {'epoch': time.time(), 'owner': owner,
                   'guard': me, 'stopping': stopping, 'shutdown_reason': shutdown_reason,
                   'active_attempt': active_attempt, 'retiring': retiring,
                   'launched': sorted(launched), 'child': child_row, 'child_exit_code': child_code,
                   'owned_process_count': len(current_own), 'pending_terminations': len(pending),
                   'remaining_seconds': max(0, deadline - now)})
        time.sleep(2)
    # No watchdog or compute process is intentionally left beyond the lease.
    for row in list(own.values()):
        if row['pid'] != me['pid']:
            signal_identity(row, signal.SIGKILL)
    remaining = [row for row in own.values() if row['pid'] != me['pid'] and alive(row)]
    write_json(node / 'guard-exit.json', {'epoch': time.time(), 'reason': shutdown_reason or 'deadline',
                                        'remaining_owned': remaining, 'npu': npu_snapshot()})
    event('guard_exit', remaining_owned=len(remaining))


if __name__ == '__main__':
    main()
