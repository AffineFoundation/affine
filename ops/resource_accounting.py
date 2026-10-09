def cgroup_headroom(maximum,current,stats):
    """Conservative automatic-reclaim allowance; never count active file pages.

    The charged cgroup file cache can exceed new anonymous-state requirements.
    Only clean inactive file pages count. Shmem belongs to the anon LRU, so
    exclude it from the type-based file envelope, not again from inactive_file.
    Mapped, dirty, writeback and unevictable pages are conservatively subtracted.
    No cache eviction is requested and no pinned page is assumed free.
    """
    if (type(maximum)is not int or maximum<0 or type(current)is not int or current<0 or
            not isinstance(stats,dict)or any(type(v)is not int or v<0 for v in stats.values())):
        raise ValueError('actual nonnegative cgroup memory accounting')
    excluded=sum(stats.get(k,0)for k in ('file_dirty','file_writeback','file_mapped','unevictable'))
    filesystem_file=max(0,stats.get('file',0)-stats.get('shmem',0))
    eligible=max(0,min(stats.get('inactive_file',0),filesystem_file)-excluded)
    # Do not exceed charged usage or the finite cgroup limit.
    eligible=min(eligible,current,maximum)
    return dict(hard_headroom_bytes=max(0,maximum-current),
        conservative_clean_inactive_file_bytes=eligible,
        usable_bytes=min(maximum,max(0,maximum-current)+eligible),
        active_file_cache_counted=False,drop_caches_requested=False)
