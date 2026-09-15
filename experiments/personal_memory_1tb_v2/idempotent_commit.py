"""Opt-in bounded retry adapter. Requires exclusive Store ownership.

Operation mapping commits inside the SAME SQLite transaction as its version.
No guarantee survives loss/rollback of that database or bypass of this adapter.
"""
import hashlib
import json

import recoverable_commit as recovery


def commit_once(store, root, operation_id, name, chunks, *, compress=True,
                profile='blocks-v1', fault=lambda stage: None):
    if not isinstance(operation_id,str) or not 1 <= len(operation_id.encode()) <= 256:
        raise ValueError('invalid operation identifier')
    if not isinstance(name,str) or len(name.encode()) > 4096 or type(compress) is not bool:
        raise ValueError('invalid write options')
    if not isinstance(profile,str) or len(profile.encode())>256 or store.db.in_transaction:
        raise ValueError('invalid profile or active transaction')
    blocks=[];total=0
    for block in chunks:
        if type(block) is not bytes or not 0 < len(block) <= 4*1024**2:
            raise ValueError('invalid block')
        total+=len(block);blocks.append(block)
        if total>64*1024**2 or len(blocks)>10000:
            raise ValueError('bounded retry input exceeded')
    key=hashlib.sha256(operation_id.encode()).hexdigest()
    signature=hashlib.sha256(json.dumps(dict(name=name,profile=profile,compress=compress,
        blocks=[(len(b),hashlib.sha256(b).hexdigest()) for b in blocks]),
        sort_keys=True,separators=(',',':')).encode()).hexdigest()
    root=recovery._root(store,root)
    if not (root/'HEAD').exists():recovery.publish(store,root)
    db=store.db
    db.execute('CREATE TABLE IF NOT EXISTS retry_operations_v1 (key TEXT PRIMARY KEY, signature TEXT NOT NULL, version INTEGER UNIQUE REFERENCES versions(id))')
    row=db.execute('SELECT signature,version FROM retry_operations_v1 WHERE key=?',(key,)).fetchone()
    if row and row[0]!=signature:
        raise ValueError('operation identifier reused with different request')
    if row and row[1] is not None:
        if not db.execute('SELECT 1 FROM versions WHERE id=?',(row[1],)).fetchone():
            raise ValueError('operation points to missing version')
        pin=recovery.publish(store,root,fault=fault)
        return dict(version=row[1],recovery_pin=pin,replayed=True)
    if row is None:
        db.execute('INSERT INTO retry_operations_v1 VALUES (?,?,NULL)',(key,signature))
    fault('after_intent')
    # key is a locally computed hex SHA-256, never interpolated user SQL.
    db.execute("CREATE TEMP TRIGGER retry_bind_v1 AFTER INSERT ON main.versions BEGIN "
               "UPDATE retry_operations_v1 SET version=NEW.id WHERE key='"+key+"' AND version IS NULL; END")
    try:
        version=store.put_version(name,blocks,compress=compress,profile=profile)
    finally:
        db.execute('DROP TRIGGER temp.retry_bind_v1')
    fault('after_store_commit')
    row=db.execute('SELECT version FROM retry_operations_v1 WHERE key=?',(key,)).fetchone()
    if row is None or row[0]!=version:
        raise ValueError('operation binding failed')
    pin=recovery.publish(store,root,fault=fault)
    return dict(version=version,recovery_pin=pin,replayed=False)
