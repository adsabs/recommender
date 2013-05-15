import os

#_basedir = os.path.abspath(os.path.dirname(__file__))
_basedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class AppConfig(object):
    
    LOG_DIR = os.path.exists(_basedir + "/logs") and _basedir + "/logs" or "."

    MONGO_HOST = "localhost"
    MONGO_SAFE = True

    SOLR_URL = 'http://adswhy:9000/solr/collection1/select'
try:
    from local_config import LocalConfig
except ImportError:
    LocalConfig = type('LocalConfig', (object,), dict())
    
for attr in filter(lambda x: not x.startswith('__'), dir(LocalConfig)):
    setattr(AppConfig, attr, LocalConfig.__dict__[attr])
    
config = AppConfig
