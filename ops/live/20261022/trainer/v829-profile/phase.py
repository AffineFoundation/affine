import csv,datetime as D,statistics as S
def ts(s): return D.datetime.strptime(s.strip(),'%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=D.timezone.utc).timestamp()
def T(h): return D.datetime(2026,10,22,*map(int,h.split(':')),tzinfo=D.timezone.utc).timestamp()
nv=[]
for r in csv.reader(open('nvsmi-e81.csv')):
    nv.append((ts(r[0]),float(r[1].split()[0]),float(r[2].split()[0]),float(r[3].split()[0]),float(r[4].split()[0]),float(r[5].split()[0])))
host=[list(map(float,l.split())) for l in open('host-e81.txt') if len(l.split())==9]
ph=[('idle before job','01:41:00','02:10:54'),('stage/load','02:10:54','02:12:49'),('KL-ref pass','02:12:49','02:15:45'),
    ('grad microbatches','02:15:45','02:32:20'),('optimizer step','02:32:20','02:33:00'),('save+publish','02:33:00','02:37:59'),('after','02:37:59','02:39:30')]
print('%-18s %6s %5s %5s %6s %6s %6s | %5s %6s %7s %7s'%('phase','sec','util','memU','memGB','W','SMMHz','cpu%','load','netTxMB/s','diskWMB/s'))
for n,a,b in ph:
    a,b=T(a),T(b); x=[r for r in nv if a<=r[0]<b]; h=[r for r in host if a<=r[0]<b]
    if not x: continue
    m=lambda i: S.mean(r[i] for r in x)
    cpu=net=dsk=float('nan'); ld=float('nan')
    if len(h)>1:
        h0,h1=h[0],h[-1]; dt=h1[0]-h0[0]
        cpu=100*(h1[2]-h0[2])/((h1[2]-h0[2])+(h1[3]-h0[3]))
        net=(h1[5]-h0[5])/dt/1e6; dsk=(h1[7]-h0[7])*512/dt/1e6; ld=S.mean(r[1] for r in h)
    print('%-18s %6.0f %5.0f %5.0f %6.1f %6.0f %6.0f | %5.0f %6.1f %7.0f %7.0f'%(n,b-a,m(1),m(2),m(3)/1024,m(4),m(5),cpu,ld,net,dsk))
g=[r for r in nv if T('02:15:45')<=r[0]<T('02:32:20')]
print('grad phase: samples',len(g),'util<50%:',sum(r[1]<50 for r in g),'power p10/p50/p90',[round(sorted(r[4] for r in g)[int(len(g)*q)]) for q in (.1,.5,.9)])
