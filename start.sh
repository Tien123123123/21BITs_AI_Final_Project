#!/bin/bash
python -m kafka_server.consumer &
python -m flask run 