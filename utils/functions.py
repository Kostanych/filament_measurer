import cv2


def nothing(args): pass
def krutilki(filepath):
    # создаем окно для отображения результата и бегунки
    cv2.namedWindow("setup")
    cv2.createTrackbar("b1", "setup", 0, 255, nothing)
    cv2.createTrackbar("g1", "setup", 0, 255, nothing)
    cv2.createTrackbar("r1", "setup", 0, 255, nothing)
    cv2.createTrackbar("b2", "setup", 255, 255, nothing)
    cv2.createTrackbar("g2", "setup", 255, 255, nothing)
    cv2.createTrackbar("r2", "setup", 255, 255, nothing)

    img = cv2.imread(filepath)  # загрузка изображения

    while True:
        r1 = cv2.getTrackbarPos('r1', 'setup')
        g1 = cv2.getTrackbarPos('g1', 'setup')
        b1 = cv2.getTrackbarPos('b1', 'setup')
        r2 = cv2.getTrackbarPos('r2', 'setup')
        g2 = cv2.getTrackbarPos('g2', 'setup')
        b2 = cv2.getTrackbarPos('b2', 'setup')
        # собираем значения из бегунков в множества
        min_p = (g1, b1, r1)
        max_p = (g2, b2, r2)
        # применяем фильтр, делаем бинаризацию
        img_g = cv2.inRange(img, min_p, max_p)

        cv2.imshow('img', img_g)

        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def krutilki_grayscale(filepath):
    cv2.namedWindow("setup")
    cv2.createTrackbar("g1", "setup", 0, 255, nothing)
    cv2.createTrackbar("g2", "setup", 0, 255, nothing)

    img = cv2.imread(filepath)  # загрузка изображения

    while True:
        g1 = cv2.getTrackbarPos('g1', 'setup')
        g2 = cv2.getTrackbarPos('g2', 'setup')

        # собираем значения из бегунков в множества
        min_p = (g1)
        max_p = (g2)
        # применяем фильтр, делаем бинаризацию
        img_g = cv2.inRange(img, min_p, max_p)

        cv2.imshow('img', img_g)

        if cv2.waitKey(33) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()