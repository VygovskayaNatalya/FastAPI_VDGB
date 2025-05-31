# FastAPI YOLO11_seg Frontend Template

Заготовка фронтенда для системы анализа изображений на базе FastAPI с заглушкой модели YOLO11_seg.

## 📋 Описание

Этот проект представляет собой готовый шаблон веб-приложения для загрузки и анализа изображений. В настоящее время использует заглушку модели YOLO11_seg, которую можно легко заменить на реальную модель.

## 🚀 Возможности

- **Двухстраничный интерфейс**: титульная страница и страница загрузки
- **Drag & Drop загрузка**: перетаскивание файлов в область загрузки
- **Предварительный просмотр**: отображение загруженного изображения
- **Заглушка YOLO11_seg**: готовая структура для подключения реальной модели
- **Responsive дизайн**: адаптивный интерфейс
- **REST API**: готовые эндпоинты для интеграции

## 📁 Структура проекта

```
FastAPI_YOLO11_Frontend/
├── main.py                 # Основной файл FastAPI приложения
├── templates/              # HTML шаблоны
│   ├── index.html         # Титульная страница
│   └── predict.html       # Страница загрузки файлов
├── static/                # Статические файлы
│   ├── style.css         # Стили 
│   └── upload.png        # Иконка загрузки
├── requirements.txt       # Зависимости Python
└── README.md             # Этот файл
```

## 🛠 Установка и запуск

### Предварительные требования

- Python 3.8+
- pip

### Установка

1. **Клонируйте репозиторий:**
```bash
git clone https://github.com/VygovskayaNatalya/FastAPI_Scolios_Spond.git
cd FastAPI_Scolios_Spond
```

2. **Создайте виртуальное окружение:**
```bash
python -m venv venv
```

3. **Активируйте виртуальное окружение:**

Windows:
```bash
venv\Scripts\activate
```

Linux/Mac:
```bash
source venv/bin/activate
```

4. **Установите зависимости:**
```bash
pip install -r requirements.txt
```

### Запуск приложения

```bash
python main.py
```

Или через uvicorn:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Приложение будет доступно по адресу: http://localhost:8000

## 📋 Зависимости

```txt
fastapi==0.104.1
uvicorn==0.24.0
python-multipart==0.0.6
jinja2==3.1.2
pillow==10.0.1
torch==2.1.0
```

## 🔧 API Эндпоинты

### GET /
Титульная страница приложения

### GET /predict
Страница загрузки файлов

### POST /predict
Анализ загруженного изображения
- **Параметры**: `image` (файл)
- **Ответ**: JSON с результатами анализа

### POST /uploadfile/
Альтернативный эндпоинт для загрузки файлов

## 🎯 Использование

1. Откройте http://localhost:8000
2. Нажмите "НАЧАТЬ РАБОТУ"
3. Загрузите изображение (drag & drop или кнопка "ВЫБРАТЬ")
4. Нажмите "РАСПОЗНАТЬ"
5. Получите результаты анализа

## 🔄 Подключение реальной модели YOLO11_seg

Для замены заглушки на реальную модель YOLO11_seg:

1. **Установите ultralytics:**
```bash
pip install ultralytics
```

2. **Замените заглушку в main.py:**
```python
from ultralytics import YOLO

# Вместо заглушки:
model = YOLO('yolo11s-seg.pt')
```

3. **Обновите функцию predict:**
```python
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    # Сохранение временного файла
    temp_file = f"temp_{image.filename}"
    with open(temp_file, "wb") as buffer:
        buffer.write(await image.read())
    
    # Предсказание
    results = model(temp_file)
    
    # Обработка результатов...
```

## 🎨 Кастомизация

### Изменение стилей
Отредактируйте CSS в файлах `templates/index.html` и `templates/predict.html`

### Добавление новых страниц
1. Создайте HTML шаблон в папке `templates/`
2. Добавьте роут в `main.py`

### Изменение логики обработки
Модифицируйте функцию `predict()` в `main.py`

## 🐛 Известные ограничения

- Заглушка модели возвращает фиктивные результаты
- Поддерживаются только изображения
- Нет сохранения результатов в базу данных
- Отсутствует аутентификация

## 🔮 Планы развития

- [ ] Подключение реальной модели YOLO11_seg
- [ ] Добавление базы данных для сохранения результатов
- [ ] Система пользователей и аутентификация
- [ ] Поддержка batch обработки
- [ ] Экспорт результатов в Excel/PDF
- [ ] API документация (Swagger)

## 📝 Лицензия

MIT License

## 👥 Авторы

- **Natalya Vygovskaya** - [VygovskayaNatalya](https://github.com/VygovskayaNatalya)

## 🤝 Вклад в проект

1. Fork проекта
2. Создайте feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit изменения (`git commit -m 'Add some AmazingFeature'`)
4. Push в branch (`git push origin feature/AmazingFeature`)
5. Откройте Pull Request

## 📞 Поддержка

Если у вас есть вопросы или предложения, создайте Issue в репозитории.

---

**Статус проекта**: 🚧 В разработке (заглушка модели)

**Последнее обновление**: май 2025