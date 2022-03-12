class Describe(object):
    TYPE_MODEL = 0
    TYPE_PREDICT = 1
    TYPE_VALIDATE = 2

    def __init__(self):
        pass

    def record(self, target_name=None) -> str:
        user_in = ''
        info = []

        def delete(history: []) -> []:
            length = len(history)

            for i in range(length):
                print(f'[{i}]:  {history[i]}')
            print('sys info> Don\'t want delete type -1')
            delete_line = int(input('kill which line:'))

            if length > delete_line >= 0:
                history.pop(delete_line)
            elif delete_line < 0:
                print('sys info > cancel delete data')
            else:
                print('input not effect')
            return history

        def show_info(history: []):
            length = len(history)

            for i in range(length):
                print(f'[{i}]: {history[i]}')
            input('type any button to continue...')

        def group(history: []) -> str:
            list_to_str = ''

            for line in history:
                list_to_str += line + "\n"
            return list_to_str

        if target_name is not None:
            print(f'sys info> file name is {target_name.split(",")[0]}')

        print('sys info> using \"!w\" discard input')
        print('sys info> using \"!del\" mean you want delete some line')
        print('sys info> using \"!see\" check what you type')
        while True:
            user_in = str(input('> '))

            if user_in == '!w':
                break
            elif user_in == '!del':
                info = delete(info)
            elif user_in == '!see':
                show_info(info)
            else:
                info.append(user_in)

        return group(info)
