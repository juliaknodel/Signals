# Большая лабораторная

## Базовая постановка задачи:

Выбранные объекты:
+ Объект А - арка в форме буквы П, однотонная

+ Объект Б - любой прямоугольный объект, однотонный

Требования к фото на входе:
+ Формат: png, jpg

+ Минимальное разрешение: 800x600

+ Освещение: искусственное или естественное, без резких теней, на фото не должно быть пересвеченных или абсолютно черных областей.

+ Не шумное, резкое, не смазанное, объекты А и Б должны быть на 100% в фокусе.

+ На фото должны находиться два объекта А и Б, которые не пересекаются и не соприкасаются, полностью помещаются на фото, между объектами и границами фотографии есть зазоры. 

Допущения вне зависимости от пришедшей на вход фотографии:
+ Считается, что объекты А и Б находятся на одинаковом удалении от камеры. 
  > Как рассчитывается расстояние: камера принимается за точку. Расстоянием считается кратчайший путь от камеры до точки принадлежащей объекту.

+ Считается, что оба объекта А и Б плоские и находятся в одной плоскости, параллельной той, в которой сделана фотография.

+ Считается, что арка стоит на полу, то есть нижняя граница арки лежит в плоскости пола.



Задача:
На вход подается фото, удовлетворяющее критериям описанным выше.

Требуется: 
+ Распознать на фото объекты А и Б == выделить четырехугольную область В внутри объекта А, в которую могут проходить другие объекты (3 края данной области описываются границами арки А, а четвертая граница полом). 

+ Установить пройдет ли объект Б под арку А (иными словами, поместится ли он в области В), при параллельном переносе объекта Б.

## Дополнительное задание:
> Указать, на сколько градусов по часовой стрелке надо повернуть объект Б относительно его центра, чтобы после параллельного переноса он влез в арку, если это возможно. За центр объекта Б принимается точка пересечения диагоналей найденного четырехугольника (в терминах пикселей: средний пиксель из занимаемых объектом по горизонтальной оси и аналогично средний по вертикальной оси).


