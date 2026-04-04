#!/bin/bash

docker kill $(docker ps | grep miniswe | awk '{print $1}')
