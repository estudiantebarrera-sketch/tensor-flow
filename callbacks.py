from tensorflow.keras.callbacks import Callback

class TrainingCallback(Callback):
    def on_train_begin(self, logs=None):
        print("Iniciando entrenamiento...")

    def on_epoch_begin(self, epoch, logs=None):
        print(f"Iniciando época {epoch + 1}/20")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        accuracy = logs.get("accuracy", 0)
        #val_accuracy = logs.get("val_accuracy", 0)

        if accuracy >= 0.95:
            print(
                "¡El modelo alcanzó el 95 % de accuracy! "
                "Deteniendo el entrenamiento."
            )
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        print("Entrenamiento finalizado.")


class TestingCallback(Callback):
    def on_test_begin(self, logs=None):
        print("Iniciando evaluación...")

    def on_test_end(self, logs=None):
        print("Evaluación finalizada.")


class PredictionCallback(Callback):
    def on_predict_begin(self, logs=None):
        print("Iniciando predicciones...")

    def on_predict_end(self, logs=None):
        print("Predicciones finalizadas.")
