# План обучения для многузловой системы (черновик)

## 1. Определить роли узлов и бюджеты
- Узлы по модальностям: аудио (KWS) и/или изображение.
- Бюджеты каждого узла: RAM/Flash/latency/энергия.

## 2. Базовый пайплайн (контрольная точка)
- Запустить micro_speech (TFLite Micro) как базовый KWS пример.
- Это дает проверенную цепочку: данные -> модель -> TFLite Micro -> ESP32.

## 3. Данные
- Базовый датасет для KWS: Speech Commands.
- Дополнительно: доменные записи и шумы для адаптации.

## 4. Многомодельность
- Обучить «teacher» на полном датасете.
- Дистиллировать компактные «student» модели под каждый узел (разные бюджеты/датчики).

## 5. (Опционально) Федеративное обучение
- Если данные нельзя централизовать, использовать FedAvg для агрегации локальных обновлений.

## 6. Квантование
- Старт: post-training full-integer quantization (нужен representative dataset).
- Если точность падает -> QAT.

## 7. Деплой и измерения
- Конверсия в TFLite / TFLite Micro.
- Измерить accuracy, latency, RAM/Flash, энергопотребление на ESP32.
- Логировать результаты в notes/Journal и notes/Research.

---

# Источники
- Speech Commands dataset (Warden, 2018): https://arxiv.org/abs/1804.03209
- TFLite Micro example micro_speech: https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/micro_speech
- micro_speech training README: https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/examples/micro_speech/train/README.md
- Distillation (Hinton et al., 2015): https://arxiv.org/abs/1503.02531
- Federated Learning / FedAvg (McMahan et al., 2017): https://arxiv.org/abs/1602.05629
- TFLite Post-Training Quantization: https://www.tensorflow.org/model_optimization/guide/quantization/post_training
- TFLite Quantization Aware Training: https://www.tensorflow.org/model_optimization/guide/quantization/training
