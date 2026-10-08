from load import Load
from utils import Utils

if __name__ == "__main__":

    load = Load()
    utils = Utils()

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
    print(classes)



