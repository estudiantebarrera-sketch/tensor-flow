from load import Load
from utils import Utils
from models import Models
from callbacks import TrainingCallback
import kerastuner as kt

if __name__ == "__main__":

    load = Load()
    utils = Utils()
    models = Models()
    callback = TrainingCallback()

    #url_json = "./in/databasesLoadData/sign_mnist_json/data.json"
    #data = utils.load_from_json(url_json)

    #url_base64 = "./in/databasesLoadData/sign_mnist_base64/data.json"
    #data = utils.load_from_base64(url_base64)

    #url_csv = "./in/databasesLoadData/sign_mnist_train/sign_mnist_train.csv"
    #data = load.load_from_csv(url_csv)
    #X, y = utils.features_target(data, ['label'], ['label'])

    #url_csv = "./in/databasesLoadData/sign_mnist_train/sign_mnist_train_clean.csv"
    #data = load.load_from_csv(url_csv)
    #X, y = utils.features_target(data, ['label'], ['label'])

    #url_train = "./in/databasesLoadData/sign_mnist_train/sign_mnist_train_clean.csv"
    #data = load.load_from_csv(url_train)
    #utils.normalizar(data, 'label')

    train_dir = "./in/sign-language-img/Train"
    test_dir = "./in/sign-language-img/Test"

    train_generator, validation_generator, test_generator, classes = load.normalizar(train_dir, test_dir)
    #history_model_base = models.model_base(train_generator, validation_generator, test_generator, classes)
    #history_model_optimizer = models.model_optimizer(train_generator, validation_generator, test_generator, classes)
    #history_model_convolutional = models.model_convolutional(train_generator, validation_generator, test_generator, classes)

    history_model_calbacks = models.model_callback(
        train_generator,
        validation_generator,
        test_generator,
        classes,
        callback
    )
    utils.visualizacion_resultados(history_model_calbacks, "model_calbacks")
    
    """tuner = kt.Hyperband(
        hypermodel=lambda hp: models.constructor_modelos(hp, classes),
        objective = "val_accuracy",
        max_epochs = 20,
        factor = 3,
        directory = "models/",
        project_name = "platzi-tunner"
    )

    tuner.search(train_generator, epochs =20, validation_data = validation_generator)

    best_hps = tuner.get_best_hyperparameters(num_trials =1)[0]

    print(best_hps.get("units"))
    print(best_hps.get("learning_rate"))

    hypermodel = tuner.hypermodel.build(best_hps)

    history_hypermodel = hypermodel.fit(
        train_generator,
        epochs = 20,
        callbacks = [callback],
        validation_data = validation_generator
    )
    utils.visualizacion_resultados(history_hypermodel, "model_hypermodel")"""





