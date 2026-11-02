#!/usr/bin/env python3
"""
快速内存回收 —— reclaim_memory.py 的加速版

目标与旧脚本相同：把 buff/cache（page cache）尽快还回去，避免训练过程中
host 内存不足（acltMallocHost / aclrtMallocHostWithCfg OOM）导致任务崩掉。

两种方式：
  1. drop  —— 直接写 /proc/sys/vm/drop_caches，瞬间释放可回收缓存（推荐）
  2. alloc —— 快速抢占内存制造压力，逼内核回收 page cache（无 root 权限时用）
              用 mmap + MADV_POPULATE_WRITE，实测 ~40GB/s；
              旧脚本 bytearray + b'\x01' 全量填充只有 ~0.7GB/s。

用法：
  python3 reclaim_memory_fast.py                     # 自动：先 drop_caches，几秒内完成
  python3 reclaim_memory_fast.py --mode alloc 1300   # 兼容旧用法：抢占 1300GB 再释放
  python3 reclaim_memory_fast.py --watch 30          # 每 30s 回收一次（跑训练时挂着）
  python3 reclaim_memory_fast.py --target-avail 1200 # 回收直到 Available >= 1200GB
"""

import argparse
import gc
import mmap
import os
import sys
import time

GB = 1024 ** 3
MADV_POPULATE_WRITE = 23  # include/uapi/linux/mman.h, Linux >= 5.14
PAGE = os.sysconf("SC_PAGE_SIZE")
DROP_CACHES = "/proc/sys/vm/drop_caches"


# --------------------------------------------------------------------------- #
# 内存状态
# --------------------------------------------------------------------------- #
def snapshot():
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            key, val = line.split(":", 1)
            info[key] = int(val.split()[0]) * 1024  # kB -> bytes
    return {
        "total": info["MemTotal"],
        "free": info["MemFree"],
        "available": info["MemAvailable"],
        "buffers": info["Buffers"],
        "cached": info["Cached"],
        "dirty": info["Dirty"],
    }


def gb(n):
    return f"{n / GB:,.1f}"


def report(title, s):
    print(
        f"{title}  total {gb(s['total'])}G | "
        f"free {gb(s['free'])}G | avail {gb(s['available'])}G | "
        f"cached {gb(s['cached'])}G | dirty {gb(s['dirty'])}G"
    )


def flush_dirty():
    """dirty page 无法被 drop，必须先落盘。正常情况几十 ms 级别。"""
    t = time.monotonic()
    os.sync()
    dt = time.monotonic() - t
    if dt > 1:
        print(f"  sync 落盘耗时 {dt:.1f}s")


# --------------------------------------------------------------------------- #
# 方式一：drop_caches
# --------------------------------------------------------------------------- #
def drop_caches():
    flush_dirty()
    try:
        with open(DROP_CACHES, "w") as f:
            f.write("3")
        return True
    except (PermissionError, OSError) as e:
        print(f"  drop_caches 不可用（{e.__class__.__name__}: {e}），改用 alloc 方式")
        return False


# --------------------------------------------------------------------------- #
# 方式二：快速内存压力
# --------------------------------------------------------------------------- #
def _populate(m):
    try:
        m.madvise(MADV_POPULATE_WRITE)
    except (AttributeError, OSError):
        # 内核 < 5.14 或 madvise 不支持时退化为逐页触碰（每页写 1 字节即可
        # 触发整页物理分配，不需要像旧脚本那样写满整个 chunk）
        for off in range(0, m.size(), PAGE):
            m[off] = 1


def allocate_pressure(target, chunk, stop_avail=None, quiet=False):
    """
    target     : 抢占的目标字节数
    chunk      : 单次 mmap 的块大小
    stop_avail : 达到该 MemAvailable 后提前结束
    返回实际抢占的字节数；内存全部在退出前释放。
    """
    chunks = []
    allocated = 0
    t0 = time.monotonic()
    try:
        while allocated < target:
            n = min(chunk, target - allocated)
            m = mmap.mmap(
                -1, n,
                flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )
            try:
                _populate(m)
            except Exception:
                m.close()
                raise
            chunks.append(m)
            allocated += n

            s = snapshot()
            if not quiet:
                speed = allocated / GB / max(time.monotonic() - t0, 1e-9)
                print(
                    f"\r  占用 {gb(allocated)}/{gb(target)}G "
                    f"({100.0 * allocated / target:5.1f}%)  "
                    f"avail {gb(s['available'])}G  cached {gb(s['cached'])}G  "
                    f"{speed:.0f}GB/s ",
                    end="",
                    flush=True,
                )
            if stop_avail is not None and s["available"] >= stop_avail:
                break
    except MemoryError:
        if not quiet:
            print(f"\n  ⚠️  内存抢占到 {gb(allocated)}G 时失败，先释放已分配的")
    finally:
        if not quiet and allocated:
            print()
        for m in chunks:
            m.close()
        chunks.clear()
        gc.collect()
    dt = time.monotonic() - t0
    if not quiet:
        print(f"  抢占 {gb(allocated)}G 用时 {dt:.1f}s，已全部释放")
    return allocated


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def reclaim(args):
    before = snapshot()
    report("回收前:", before)

    if args.mode in ("auto", "drop"):
        ok = drop_caches()
        s = snapshot()
        d = s["cached"] - before["cached"]
        print(
            f"  drop_caches 完成：cached {gb(before['cached'])}G -> {gb(s['cached'])}G"
            f"（{'+' if d >= 0 else '-'}{gb(abs(d))}G），"
            f"free {gb(before['free'])}G -> {gb(s['free'])}G"
        )
        if not ok and args.mode == "drop":
            sys.exit(1)

    # alloc 方式：显式指定，或 auto 下 drop 之后仍未达到 --target-avail
    need_alloc = args.mode == "alloc"
    if args.mode == "auto" and args.target_avail:
        need_alloc = snapshot()["available"] < args.target_avail * GB

    if need_alloc:
        cur = snapshot()
        chunk = max(int(args.chunk * GB), PAGE)
        if args.target is not None:
            target = int(args.target * GB)
        else:
            # 与旧脚本一致的保守默认：最多抢到当前可用内存的 90%
            target = max(int(cur["available"] * 0.9), chunk)
        stop = int(args.target_avail * GB) if args.target_avail else None
        print(f"  内存压力模式：抢占 {gb(target)}G（每次 {gb(chunk)}G）")
        allocate_pressure(target, chunk, stop_avail=stop, quiet=args.quiet)

    after = snapshot()
    report("回收后:", after)
    freed_cache = (before["cached"] + before["buffers"]) - (after["cached"] + after["buffers"])
    sign = "+" if freed_cache >= 0 else "-"
    print(
        f"✅ 缓存释放 {sign}{gb(abs(freed_cache))}G；"
        f"free {gb(before['free'])}G -> {gb(after['free'])}G "
        f"（共享机器上 free 还会被其他进程影响）"
    )


def main():
    p = argparse.ArgumentParser(
        description="快速回收 buff/cache（drop_caches + 高速内存压力）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("target", nargs="?", type=float,
                   help="兼容旧脚本：要抢占的内存 GB 数（给了就自动用 alloc 模式）")
    p.add_argument("--mode", choices=["auto", "drop", "alloc"], default="auto",
                   help="auto=先 drop_caches，不够再抢内存；drop=只清缓存；alloc=只抢内存")
    p.add_argument("--chunk", type=float, default=16,
                   help="alloc 模式单次 mmap 大小 GB")
    p.add_argument("--target-avail", type=float, default=None, metavar="GB",
                   help="回收直到 MemAvailable 达到该值")
    p.add_argument("--watch", type=int, default=0, metavar="SEC",
                   help="循环执行，每 SEC 秒回收一次（0=只跑一次）")
    p.add_argument("--quiet", action="store_true", help="精简输出")
    args = p.parse_args()

    if args.target is not None and args.mode == "auto":
        args.mode = "alloc"

    banner = "内存快速回收" + (f"（每 {args.watch}s 一次，Ctrl+C 退出）" if args.watch else "")
    print(banner)
    try:
        while True:
            t = time.monotonic()
            reclaim(args)
            if not args.watch:
                break
            sleep = max(args.watch - (time.monotonic() - t), 0)
            time.sleep(sleep)
    except KeyboardInterrupt:
        print("\n已中断")


if __name__ == "__main__":
    main()