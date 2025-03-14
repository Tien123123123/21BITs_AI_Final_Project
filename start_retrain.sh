#!/bin/bash
python -m kafka_server.retrain_consumer &
python -m flask run 