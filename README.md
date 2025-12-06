
# Flask API for RandomForest Model

Небольшой Flask-сервис, который загружает `randomforest_model.joblib` (положите рядом с `app.py`) и предоставляет два эндпоинта:

- `POST /predict` — принимает JSON с признаками и возвращает предсказание и вероятность.
- `GET /meta` — возвращает информацию о модели и порядке признаков.

Установка и запуск

1) Установите зависимости (PowerShell):

```powershell
python -m pip install -r requirements.txt
```

2) Запустите API (PowerShell):

```powershell
python app.py
```

Формат входных данных

Тело запроса в `POST /predict` должно быть JSON-объектом с ключами — названиями признаков. Все признаки обязательны, порядок определяется моделью (через `feature_names_in_`) или следующей последовательностью по умолчанию:

```
region, resource_type, water_type, fauna, passport_age, is_old_passport,
technical_condition, condition_inv, water_level_upstream, water_level_downstream,
level_diff, exploitation_stress, height, width, depth, num_incidents_past,
latitude, longitude
```

Пример тела запроса (payload.json):

```json
{
  "region": 1,
  "resource_type": 0,
  "water_type": 2,
  "fauna": 0,
  "passport_age": 12,
  "is_old_passport": 0,
  "technical_condition": 2,
  "condition_inv": 1,
  "water_level_upstream": 3.2,
  "water_level_downstream": 2.9,
  "level_diff": 0.3,
  "exploitation_stress": 1,
  "height": 5.0,
  "width": 2.3,
  "depth": 1.5,
  "num_incidents_past": 0,
  "latitude": 55.7558,
  "longitude": 37.6176
}
```

Примеры запросов

- curl (если у вас доступен `curl.exe`):

```powershell
curl.exe -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d @payload.json
```

- PowerShell (Invoke-RestMethod):

```powershell
Invoke-RestMethod -Uri http://localhost:5000/predict -Method Post -Body (Get-Content .\payload.json -Raw) -ContentType 'application/json'
```

- Python (requests):

```python
import requests, json
url = 'http://localhost:5000/predict'
with open('payload.json', 'r', encoding='utf-8') as f:
    payload = json.load(f)
resp = requests.post(url, json=payload)
print(resp.status_code, resp.json())
```

Пример ответа

```json
{
  "prediction": 1,
  "probability": 0.8423,
  "used_features": ["region", "resource_type", "water_type", ...]
}
```

Эндпойнт `/meta`

```powershell
Invoke-RestMethod -Uri http://localhost:5000/meta -Method Get
```

Вернёт JSON с информацией о загруженной модели, числе признаков и порядке признаков, которые использует API.

Важные замечания

- Если модель обучалась на закодированных категориальных признаках (например, использовались `cat.codes`), API ожидает такие же целочисленные коды в запросе. Если вы хотите, чтобы API принимал текстовые значения и сам выполнял маппинг, пришлите файл подготовки данных (CSV) или маппинг кодов — я добавлю автоматическое преобразование.
- Для локального тестирования встроенный сервер Flask (в `app.py`) подходит, но для продакшна используйте WSGI-сервер (`gunicorn`, `waitress` и т.п.).

Если хотите, могу дополнительно:
- добавить валидацию типов и диапазонов входных значений;
- реализовать автоматический маппинг категорий по `prepared_dataset.csv` или исходному CSV;
- добавить пример клиента на FastAPI или Dockerfile для контейнера.

