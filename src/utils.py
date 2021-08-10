def log(message):
    with open('logs.txt', 'a') as file:
        file.write(str(message) + '\n')