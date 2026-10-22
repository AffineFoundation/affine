p='/home/const/subnet120-rewrite/state/root-audits/20261021-v827-incremental-capture/activate_v827.sh';s=open(p).read()
a='n=$(ls "$E/coordinator/captured" 2>/dev/null | wc -l)'
assert s.count(a)==2, s.count(a)
s=s.replace(a,'n=$( (ls "$E/coordinator/captured" 2>/dev/null || true) | wc -l)')
old1="""tr '\\0' ' ' < /proc/$pid/cmdline | grep -q "uid-operator.v8.2.7.ROOT-ADMITTED" &&"""
assert s.count(old1)==1,'cmd'
s=s.replace(old1,"""CMD=$(tr '\\0' ' ' < /proc/$pid/cmdline); grep -q "uid-operator.v8.2.7.ROOT-ADMITTED" <<<"$CMD" &&""")
old2="""journalctl --user -u affine-mimo-lifecycle.service --since "-2min" --no-pager | grep -q '"component": "host-v824"' &&"""
assert s.count(old2)==1,'j'
s=s.replace(old2,"""J=$(journalctl --user -u affine-mimo-lifecycle.service --since "-2min" --no-pager 2>/dev/null || true); grep -q '"component": "host-v824"' <<<"$J" &&""")
old3="""journalctl --user -u affine-mimo-lifecycle.service --since "-1min" --no-pager | tail -8 | cut -c1-300 | tee -a"""
assert s.count(old3)==1,'t'
s=s.replace(old3,"""tail -8 <<<"$J" | cut -c1-300 | tee -a""")
open(p,'w').write(s);print('patched')
