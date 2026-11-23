"""File-length/allocation and process costs, never total process-tree peak RAM."""
import hashlib,resource,stat,time

def inventory(root):
    out={}
    for p in sorted(root.rglob('*')):
        s=p.lstat()
        if stat.S_ISDIR(s.st_mode):continue
        if not stat.S_ISREG(s.st_mode):raise ValueError('nonregular retained entry')
        out[str(p.relative_to(root))]=dict(bytes=s.st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),allocated_bytes=s.st_blocks*512)
    return out

def disk(root):
    inv=inventory(root);categories={}
    for n,r in inv.items():
        category=('decoder' if n=='decoder' else 'snapshots' if n.startswith('snapshots/') else
            'staging' if n.startswith('staging/') else 'packed_indexes' if n.endswith('.packed') else
            'object_metadata' if n.startswith('objects/') else 'CURRENT_LOCK_sidecars')
        categories[category]=categories.get(category,0)+r['bytes']
    return dict(file_lengths=sum(r['bytes'] for r in inv.values()),
        file_allocated_bytes=sum(r['allocated_bytes'] for r in inv.values()),
        directory_allocated_bytes=sum(p.lstat().st_blocks*512 for p in [root,*root.rglob('*')] if p.is_dir()),
        categories=categories,files=inv)

def measured(fn):
    start=time.monotonic();s=resource.getrusage(resource.RUSAGE_SELF);c=resource.getrusage(resource.RUSAGE_CHILDREN)
    value=fn();after=resource.getrusage(resource.RUSAGE_SELF);child=resource.getrusage(resource.RUSAGE_CHILDREN)
    return value,dict(wall_seconds=time.monotonic()-start,
        self_cpu_seconds=after.ru_utime+after.ru_stime-s.ru_utime-s.ru_stime,
        reaped_children_cpu_seconds=child.ru_utime+child.ru_stime-c.ru_utime-c.ru_stime,
        self_peak_rss_kib_cumulative=after.ru_maxrss,
        max_reaped_child_rss_kib_cumulative=child.ru_maxrss,process_tree_peak_rss_measured=False)
