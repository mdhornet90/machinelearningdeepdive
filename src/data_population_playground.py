with open('datasets/non_linear_train') as file:
    numbers = [int(line.split()[0]) for line in file.readlines()]

with open('datasets/non_linear_train', 'w') as file:
    file.writelines(['{} {}\n'.format(number, 2 * number * number + 6 * number + 4) for number in numbers])