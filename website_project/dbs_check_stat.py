import sqlite3

connection = sqlite3.connect("statistic.db")
cursor = connection.cursor()
res = cursor.execute("select * from statistic")

cols = res.description
stats = tuple(res)[0]
print("Number of image downloads for each privacy level:\n")
for i in range(len(cols)):
    print(cols[i][0], ":", stats[i])

connection.close()
