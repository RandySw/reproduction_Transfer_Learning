
tmp = [0]

with open("predict.txt", 'w') as f:
    f.write('Id,Category\n')
    for i in enumerate(tmp):
        f.write('{}\n'.format(i))