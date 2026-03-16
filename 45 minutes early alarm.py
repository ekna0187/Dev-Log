n=int(input("Enter hour: "))
m=int(input("Enter minute: "))
if 0<=n<=23 and 0<=m<=59:
    if m<45:
        n-=1
        m+=15
    else:        m-=45
    print(str(n) + "시" + str(m) + "분")
