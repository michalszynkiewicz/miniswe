#!/bin/bash

docker kill $(docker ps | grep buzztr | awk '{print $1}')
