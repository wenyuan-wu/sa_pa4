import logging
import gensim.downloader as api

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
info = api.info()

for model_name, model_data in sorted(info['models'].items()):
    print(
        '%s (%d records): %s' % (
            model_name,
            model_data.get('num_records', -1),
            model_data['description'][:40] + '...',
        )
    )
