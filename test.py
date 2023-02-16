import PySimpleGUI as sg

sg.theme('DarkAmber')
# Устанавливаем цвет внутри окна
layout = [  [sg.Text('Некоторый текст в строке №1')],
            [sg.Text('Введите «хоть что-нибудь» в строку №2'), sg.InputText()],
            [sg.Button('Ввод'), sg.Button('Отмена')] ]

# Создаем окно
window = sg.Window('Название окна', layout)
# Цикл для обработки "событий" и получения "значений" входных данных
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Отмена':
# если пользователь закрыл окно или нажал «Отмена»
        break
    print('Молодец, ты справился с вводом', values[0])

window.close()


# sg.theme_previewer()