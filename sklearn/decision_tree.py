from sklearn import tree

#step 1: thu thap du lieu 
#step 2: xu ly du lieu
#step 3: xay dung model
#step 4: du doan ket qua
#step 5: danh gia ket qua

my_tree = tree.DecisionTreeClassifier()
input  = [[ 1, 3, 3, 7],
            [5, 2, 4, 6],
            [1, 2, 4, 6],
            [5, 4, 4, 3],
            [1, 4, 4, 7],
            [3, 2, 3, 7],
            [3, 3, 3, 6],
            [5, 2, 2, 7]
            ]

label = [0, 1, 1, 0, 0, 0, 0, 1]

result = my_tree.fit(input, label)

output = result.predict([[1, 4, 3, 6]]) 
print("Ket qua du doan la: ", output)