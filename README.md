## Основное описание

Для набора данных проекта программа предварительно обрабатывает его, загружает модели, использует их для прогнозирования аномалий, обнаруживает их и создает отчет в формате JSON.

### Системные требования 

- pandas==1.3.4
- tensorflow==2.12.0
- joblib==1.1.0
- loguru==0.5.3
- sklearn==0.22.2.post1  
- scipy==1.3.3
- clickhouse-driver==0.2.5

### Как установить
```
pip install -r requirements.txt
```
## Как использовать 

### Использование аргументов командной строки
Эта программа может принимать следующие аргументы командной строки:

- `--config_path` - путь к YAML файлу конфигурации. (по умолчанию: '')
- `--file_format` - Формат входных данных: db, csv, sqlite. (по умолчанию: db)
- `--csv_kks` - Путь к csv-файлу сконфигурированной группы kks. (по умолчанию: False)

Запустите следующую команду для запуска программы:
   
    python predict_offline.py --config_path=path/to/config --file_format=db --csv_kks=False
   
Обязательными аргументами являются путь к файлу конфигурации (`--config_path`), формат входных данных (`--file_format`), который может иметь следующие значения csv,json и SQLite, а также путь к файлу сконфигурированной группы KKS (`--csv_kks`), который может иметь значение `False` или путь к csv-файлу. 

### Пример запуска
```
python predcit_offline.py --config_path SOCHI
```


### Ожидаемые результаты

Программа обрабатывает набор данных, прогнозирует аномалии, обнаруживает их и сохраняет отчет в JSON-файл.

### Конфигурация

Программу можно настроить с помощью файла `config_offline.yaml`, который содержит следующие параметры:

`NUM_GROUPS`: это количество групп в наборе данных.

`KKS`: это файл, содержащий KKS.

`WEIGHTS`: здесь указан каталог для весов модели.

`SCALER`: здесь указан каталог для масштабов модели.

`LAG`: это задержка прогнозирования для модели.

`ROLLING_MEAN`: это логическое значение, которое позволяет скользящее среднее данных.

`EXP_SMOOTH`: это логическое значение, которое позволяет экспоненциально сглаживать данные.

`DOUBLE_EXP_SMOOTH`: это логическое значение, которое позволяет двухэтапное экспоненциально сглаживание данных.

`ROLLING_MEAN_WINDOW`: это указывает размер окна для скользящего среднего.

`ROLLING_MEAN_LOSS`: это размер скользящего окна для целевого значения.

`ANOMALY_TRESHOLD`: это требуемое пороговое значение для потерь.

`COUNT_ANOMALY`: это настройка определяет минимальную длину любого обнаруженного интервала аномалии.

`COUNT_TOP`: это количество датчиков, которые будут записаны в отчет.

`POWER_ID`: это индекс столбца мощности в наборе данных.

`POWER_LIMIT`: это задает предел для набора данных.

`MEAN_NAN`: это логическое значение, которое позволяет использовать среднее значение при предварительной обработке NA-значений в наборе данных.

`DROP_NAN`: это логическое значение, которое позволяет удалить строки, которые содержат пустые ячейки.

`AUTO_DROP_LIST`: это логическое значение, которое удаляет датчики нулевой группы из отчета csv.

`CSV_SAVE_RESULT`: это каталог, который содержит результат.

`CSV_DATA`: это каталог, который содержит исходный набор данных.

`JSON_DATA`: это каталог, который содержит выходной отчет JSON.

`TEST_FILE`: Указывается способ получения данных, возможны форматы: запрос к бд, sqlite, csv.

`DROP_LIST`: это список датчиков, которые нужно удалить из отчета.