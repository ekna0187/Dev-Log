a=int(input("Enter the first number: "))
b=int(input("Enter the second number: "))
c=int(input("Enter the third number: "))
if (a==b==c):
    print(10000+a*1000)
elif (a==b or b==c):
    print(1000+b*100) 
    if (a==c):
        print(1000+a*100)
else:
    larger=max(a,b,c)
    print(larger*100)
