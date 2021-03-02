print("commit")
print("commit2")

# 동적 변수
j = 1
for i in range (1, 4) :
 
    # inp = input("typing_vatiable_{}:".format(i))
    inp =  "variable" + str(1)
    # globals()['Var_{}'.format(i)] = inp
    globals()['Var_{}'.format(j)] = inp
    j += 1

print(Var_1, Var_2, Var_3)

variable1 = ['a','b','c']
print(variable1)
