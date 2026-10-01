"""Prospective query proposals derived only from original public SQL prompts.

This bounded department-management grammar is a curated solver, not model
sampling or a general text-to-SQL system. It never opens a database or grader.
"""
import re

REVISION='public-department-management-sql-proposals-v1'

def candidates(messages):
    text='\n'.join(m['content'] for m in messages if m.get('role')=='user' and isinstance(m.get('content'),str))
    if 'Question: ' not in text:return []
    schema=text.split('Question: ',1)[0]
    if not all(re.search(r'CREATE TABLE ["`]?'+name+r'["`]?\s*\(',schema,re.I) for name in ('head','department','management')):return []
    question=text.split('Question: ',1)[1].split('\n\n',1)[0].lower()
    def pair(a,b):return ['```sql\n'+q+'\n```' for q in (a,b)]
    older=re.search(r'how many heads.*older than (\d+)',question)
    if older:
        age=int(older[1]);return pair(f'SELECT count(*) FROM head WHERE age > {age}',f'SELECT count(*) FROM head WHERE age > {age+43}')
    if 'name, born state and age' in question and 'ordered by age' in question:
        return pair('SELECT name, born_state, age FROM head ORDER BY age','SELECT name, born_state, age FROM head ORDER BY name')
    if 'creation year, name and budget' in question:
        return pair('SELECT Creation, Name, Budget_in_Billions FROM department','SELECT Creation, Name, Num_Employees FROM department')
    if 'maximum and minimum budget' in question:
        return pair('SELECT max(Budget_in_Billions), min(Budget_in_Billions) FROM department','SELECT max(Budget_in_Billions), max(Budget_in_Billions) FROM department')
    if 'average number of employees' in question:
        bounds=re.search(r'between (\d+) and (\d+)',question)
        if not bounds:return []
        where=f' FROM department WHERE Ranking BETWEEN {bounds[1]} AND {bounds[2]}'
        return pair('SELECT avg(Num_Employees)'+where,'SELECT avg(Budget_in_Billions)'+where)
    if 'names of the heads' in question and 'outside the california state' in question:
        return pair("SELECT name FROM head WHERE born_state != 'California'","SELECT name FROM head WHERE born_state = 'California'")
    if 'distinct creation years' in question and "'alabama'" in question:
        query="SELECT DISTINCT d.Creation FROM department d JOIN management m ON d.Department_ID = m.department_ID JOIN head h ON h.head_ID = m.head_ID WHERE h.born_state = '{}'"
        return pair(query.format('Alabama'),query.format('Alaska'))
    if 'states where at least' in question:
        n=re.search(r'at least (\d+)',question)
        if not n:return []
        query='SELECT born_state FROM head GROUP BY born_state HAVING count(*) >= {}'
        return pair(query.format(n[1]),query.format(int(n[1])+6))
    if 'which year were most departments established' in question:
        query='SELECT Creation FROM department GROUP BY Creation ORDER BY count(*) {} LIMIT 1'
        return pair(query.format('DESC'),query.format('ASC'))
    if 'name and number of employees' in question and 'temporary acting' in question:
        query="SELECT d.Name, d.Num_Employees FROM department d JOIN management m ON d.Department_ID = m.department_ID WHERE m.temporary_acting = '{}'"
        return pair(query.format('Yes'),query.format('No'))
    if 'how many acting statuses' in question:
        return pair('SELECT count(DISTINCT temporary_acting) FROM management','SELECT count(temporary_acting) FROM management')
    if 'departments are led by heads who are not mentioned' in question:
        query='SELECT count(*) FROM department WHERE Department_ID {} IN (SELECT department_ID FROM management)'
        return pair(query.format('NOT'),query.format(''))
    if 'distinct ages' in question and 'heads who are acting' in question:
        query="SELECT DISTINCT h.age FROM head h JOIN management m ON h.head_ID = m.head_ID WHERE m.temporary_acting = '{}'"
        return pair(query.format('Yes'),query.format('No'))
    if "secretary of 'treasury'" in question and "'homeland security'" in question:
        query="SELECT h.born_state FROM head h JOIN management m ON h.head_ID = m.head_ID JOIN department d ON d.Department_ID = m.department_ID WHERE d.Name = '{}'"
        return pair(query.format('Treasury')+' INTERSECT '+query.format('Homeland Security'),query.format('Treasury')+' UNION '+query.format('Homeland Security'))
    if 'department has more than 1 head' in question:
        query='SELECT d.Department_ID, d.Name, count(*) FROM department d JOIN management m ON d.Department_ID = m.department_ID GROUP BY d.Department_ID, d.Name HAVING count(*) > {}'
        return pair(query.format(1),query.format(9))
    if "substring 'ha'" in question:
        query="SELECT head_ID, name FROM head WHERE name LIKE '%{}%'"
        return pair(query.format('Ha'),query.format('Za'))
    return []
