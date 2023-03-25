with open('train_sum.txt', 'r') as f:
    lines = f.readlines()
single=0
double=0
new_lines = list(lines[0])

open_d = 0
for i,c in enumerate(new_lines):
    if '[' == c:
        open_d+=1
    elif ']' == c:
        open_d-=1
    
    if open_d==3:
        if c=='\'' and double==0 and single==0:
            new_lines[i] = "\""
            single = 1
        elif c=="\"" and double==0:
            double = 1
        elif c=="\"" and double==1:
            double = 0
        elif c=='\'' and double==0 and single==1 and (new_lines[i+1]==',' or new_lines[i+1]==']'):
            new_lines[i] = "\""
            single = 0

with open('train_sum_2.txt', 'w') as f:
    f.write(''.join(new_lines))

with open('train_sum_2.json','w') as f:
    f.write(''.join(new_lines))



        



