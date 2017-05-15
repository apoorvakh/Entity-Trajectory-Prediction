import csv,os

source="Stanford drone dataset\\annotations"
destination="Stanford drone dataset\\annotations"

folders=['bookstore','coupa','deathCircle','gates','hyang','little','nexus','quad']

for folder in folders:
    num=0
    for x,y,z in os.walk(source+'\\'+folder):
    
        num=len(y)
        break
    for video in range(0,num):
        print(source+'\\'+folder+'\\video'+str(video))
        s=open(source+'\\'+folder+'\\video'+str(video)+'\\annotations.txt','r')
        d=open(destination+'\\'+folder+'\\video'+str(video)+'\\'+folder+str(video)+'.csv','w',newline='')
        writer = csv.writer(d)
        
        for line in s.readlines():
            line=line.strip()
            row=line.split(' ')
            
            
            #print type(row), row
            x_mid = (int(row[1]) + int(row[3])) / 2
            y_mid = (int(row[2]) + int(row[4])) / 2
            row.insert(5,int(x_mid))
            row.insert(6,int(y_mid))
            #print x_mid
            #print type(row), row
            
            row[-1]=row[-1].replace('"','')
            writer.writerow(row)
        s.close()
        d.close()

            
        
