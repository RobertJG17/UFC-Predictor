from kaggle.api.kaggle_api_extended import KaggleApi


def download_data():
    api = KaggleApi()
    api.authenticate()

    api.dataset_download_file(
        'mdabbert/ultimate-ufc-dataset',
        file_name='upcoming-event.csv',
    )
