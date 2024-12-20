# ohmyink-segment-anything


```
gunicorn main:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --daemon
```


with newrelic monitoring
```
NEW_RELIC_CONFIG_FILE=/PATH/TO/newrelic.ini newrelic-admin run-program gunicorn main:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --daemon
```