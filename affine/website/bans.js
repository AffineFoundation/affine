(()=>{const $=id=>document.getElementById(id);
const MON=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
const crDate=ts=>{const d=new Date((ts-6*3600)*1000);return MON[d.getUTCMonth()]+' '+d.getUTCDate()+', '+String(d.getUTCHours()).padStart(2,'0')+':'+String(d.getUTCMinutes()).padStart(2,'0')+' CR';};
const td=(t,c)=>{const e=document.createElement('td');e.textContent=t;if(c)e.className=c;return e;};
async function load(){const body=$('bans-body');if(!body)return;
 try{const r=await fetch('/bans.json?t='+Date.now(),{cache:'no-store'});if(!r.ok)throw 0;const d=await r.json();
  const rows=(d.banned||[]).map(b=>{const tr=document.createElement('tr');
   tr.append(td(b.uid==null?'—':String(b.uid)),td(b.hotkey_short||'—','hotkey-cell'),
    td((b.banned_since_epoch!=null?'E'+b.banned_since_epoch+' · ':'')+(typeof b.banned_at==='number'?crDate(b.banned_at):(b.banned_at_cr||''))),
    td((b.audits_failed||[]).map(e=>'E'+e).join(', ')||'—'),td(b.reason||'—'));
   if(b.hotkey)tr.children[1].title=b.hotkey;return tr;});
  body.replaceChildren(...rows);
  $('bans-count').textContent=String(d.count??rows.length);
  $('bans-asof').textContent=d.generated_at_cr?'As of '+d.generated_at_cr:'';
 }catch{body.innerHTML='<tr><td colspan="5">Ban list unavailable.</td></tr>';}}
if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',load);else load();
setInterval(load,300000);})();
